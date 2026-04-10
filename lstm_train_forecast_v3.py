import math
import re
import random
import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =========================================================
# 재현성
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 상수 / 파라미터
# =========================================================
CHECKIN_LETTERS = set(list("ABCDEFGHIJKLMN"))

WAIT_PARAMS = {
    "checkin": {"alpha": 4.0, "gamma": 0.09, "R": 6.0, "beta": 1.5, "wmax": 120.0},
    "security": {"alpha": 5.0, "gamma": 0.11, "R": 8.0, "beta": 2.0, "wmax": 120.0},
    "transit": {"alpha": 1.5, "gamma": 0.11, "R": 20.0, "beta": 4.0, "wmax": 60.0},
    "outside": {"alpha": 0.0, "gamma": 0.0, "R": 9999.0, "beta": 0.0, "wmax": 0.0},
}

DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})"),
]

# =========================================================
# 구역 분류 / 대기시간
# =========================================================
def normalize_area_name(area_name: str) -> str:
    return str(area_name).strip().upper()

def classify_area(area_name: str) -> str:
    a = normalize_area_name(area_name)

    if a in CHECKIN_LETTERS:
        return "checkin"

    security_keywords = ["SECURITY", "SEARCH", "SCREEN", "DEPARTURE", "GATE", "출국", "보안", "검색", "심사"]
    if any(k in a for k in security_keywords):
        return "security"

    if re.fullmatch(r"\d+", a):
        return "security"

    transit_keywords = ["GREAT", "HALL", "홀", "로비", "이동"]
    if any(k in a for k in transit_keywords):
        return "transit"

    if a == "OUTSIDE":
        return "outside"

    return "transit"

def compute_wait_time_exp(area_type: str, n_eff: float) -> float:
    p = WAIT_PARAMS[area_type]
    if area_type == "outside":
        return 0.0

    wait_min = p["beta"] + p["alpha"] * (math.exp(p["gamma"] * (n_eff / p["R"])) - 1.0)
    wait_min = min(wait_min, p["wmax"])
    return round(max(0.0, wait_min), 2)

def add_wait_time_columns(df: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    out = df.copy()
    out["area"] = out["area"].astype(str)
    out["area_type"] = out["area"].apply(classify_area)
    out = out.sort_values(["area", "time_index"]).copy()

    lag_steps_1min = max(1, 60 // interval_seconds)
    out["num_people_prev_1min"] = (
        out.groupby("area")["num_people"]
           .shift(lag_steps_1min)
           .fillna(out["num_people"])
    )

    out["n_eff"] = 0.7 * out["num_people"] + 0.3 * out["num_people_prev_1min"]
    out["wait_time_min"] = out.apply(
        lambda row: compute_wait_time_exp(row["area_type"], row["n_eff"]),
        axis=1
    )
    return out

# =========================================================
# 데이터 로드
# =========================================================
def parse_date_from_filename(path: Path):
    name = path.stem
    for pat in DATE_PATTERNS:
        m = pat.search(name)
        if m:
            y, mth, d = m.groups()
            return pd.Timestamp(f"{y}-{mth}-{d}").date()
    return None

def standardize_single_file(df_raw: pd.DataFrame, file_date):
    cols = set(df_raw.columns)

    if {"time_index", "area", "num_people"}.issubset(cols):
        df = df_raw[["time_index", "area", "num_people"]].copy()
    elif {"time_index", "area", "mac_address"}.issubset(cols):
        df = (
            df_raw.groupby(["time_index", "area"])["mac_address"]
                  .nunique()
                  .reset_index(name="num_people")
        )
    else:
        raise ValueError(
            "필수 컬럼 부족: (time_index, area, num_people) 또는 (time_index, area, mac_address)"
        )

    df["time_index"] = pd.to_numeric(df["time_index"], errors="coerce")
    df["num_people"] = pd.to_numeric(df["num_people"], errors="coerce")
    df["area"] = df["area"].astype(str)

    df = df.dropna(subset=["time_index", "num_people", "area"]).copy()
    df["time_index"] = df["time_index"].astype(int)
    df["num_people"] = df["num_people"].clip(lower=0).round().astype(int)
    df["date"] = pd.to_datetime(file_date).date()
    return df

def load_daily_files(folder_path: str, file_glob: str, raw_interval_seconds: int):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {folder_path}")

    files = sorted(folder.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"조건에 맞는 파일이 없습니다: {folder / file_glob}")

    daily_frames = []

    for fp in files:
        try:
            df_raw = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] 파일 읽기 실패: {fp.name} -> {e}")
            continue

        file_date = None
        if "date" in df_raw.columns:
            tmp = pd.to_datetime(df_raw["date"], errors="coerce").dropna()
            if not tmp.empty:
                file_date = tmp.iloc[0].date()

        if file_date is None:
            file_date = parse_date_from_filename(fp)

        if file_date is None:
            print(f"[WARN] 날짜 미인식으로 건너뜀: {fp.name}")
            continue

        try:
            df = standardize_single_file(df_raw, file_date)
            daily_frames.append(df)
        except Exception as e:
            print(f"[WARN] 형식 변환 실패: {fp.name} -> {e}")

    if not daily_frames:
        raise ValueError("사용 가능한 CSV가 없습니다.")

    all_df = pd.concat(daily_frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.date
    all_df["area"] = all_df["area"].astype(str)

    steps_per_day_raw = 86400 // raw_interval_seconds
    all_dates = sorted(all_df["date"].unique().tolist())
    all_areas = sorted(all_df["area"].unique().tolist())
    all_times = list(range(steps_per_day_raw))

    full_grid = pd.MultiIndex.from_product(
        [all_dates, all_times, all_areas],
        names=["date", "time_index", "area"]
    ).to_frame(index=False)

    all_df = (
        full_grid.merge(
            all_df.groupby(["date", "time_index", "area"], as_index=False)["num_people"].sum(),
            on=["date", "time_index", "area"],
            how="left"
        ).fillna({"num_people": 0})
    )

    all_df["num_people"] = all_df["num_people"].astype(int)
    all_df["actual_time"] = pd.to_datetime(all_df["date"]) + pd.to_timedelta(
        all_df["time_index"] * raw_interval_seconds, unit="s"
    )
    all_df["area_type"] = all_df["area"].apply(classify_area)

    return all_df, all_dates, all_areas

def aggregate_interval(df: pd.DataFrame, raw_interval_seconds: int, model_interval_seconds: int) -> pd.DataFrame:
    if model_interval_seconds % raw_interval_seconds != 0:
        raise ValueError("model_interval_seconds는 raw_interval_seconds의 배수여야 합니다.")

    ratio = model_interval_seconds // raw_interval_seconds
    out = df.copy()
    out["bucket_index"] = out["time_index"] // ratio

    agg = (
        out.groupby(["date", "bucket_index", "area"], as_index=False)["num_people"]
           .mean()
    )
    agg["num_people"] = agg["num_people"].round().astype(int)
    agg = agg.rename(columns={"bucket_index": "time_index"})
    agg["actual_time"] = pd.to_datetime(agg["date"]) + pd.to_timedelta(
        agg["time_index"] * model_interval_seconds, unit="s"
    )
    agg["area_type"] = agg["area"].apply(classify_area)
    return agg

# =========================================================
# baseline
# =========================================================
def build_baseline_for_date(train_df: pd.DataFrame, target_date, all_areas, steps_per_day):
    """
    train_df는 target_date 이전 데이터가 들어오는 것을 권장.
    비어 있어도 안전하게 동작.
    """
    target_ts = pd.to_datetime(target_date)
    target_dow = target_ts.dayofweek
    target_is_weekend = int(target_dow >= 5)

    base_grid = pd.MultiIndex.from_product(
        [[target_date], range(steps_per_day), all_areas],
        names=["date", "time_index", "area"]
    ).to_frame(index=False)

    base_grid["dow"] = target_dow
    base_grid["is_weekend"] = target_is_weekend

    if train_df is None or train_df.empty:
        base_grid["baseline_people"] = 0.0
        return base_grid[["date", "time_index", "area", "baseline_people"]]

    tmp = train_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
    tmp["date_ts"] = pd.to_datetime(tmp["date"])
    tmp["dow"] = tmp["date_ts"].dt.dayofweek
    tmp["is_weekend"] = (tmp["dow"] >= 5).astype(int)

    slot_mean = (
        tmp.groupby(["time_index", "area"], as_index=False)["num_people"]
           .mean()
           .rename(columns={"num_people": "slot_mean"})
    )

    weekend_mean = (
        tmp.groupby(["is_weekend", "time_index", "area"], as_index=False)["num_people"]
           .mean()
           .rename(columns={"num_people": "weekend_slot_mean"})
    )

    dow_mean = (
        tmp.groupby(["dow", "time_index", "area"], as_index=False)["num_people"]
           .mean()
           .rename(columns={"num_people": "dow_slot_mean"})
    )

    base = base_grid.merge(slot_mean, on=["time_index", "area"], how="left")
    base = base.merge(weekend_mean, on=["is_weekend", "time_index", "area"], how="left")
    base = base.merge(dow_mean, on=["dow", "time_index", "area"], how="left")

    train_dates_sorted = sorted(tmp["date"].unique().tolist())
    if len(train_dates_sorted) > 0:
        prev_date = train_dates_sorted[-1]
        prev_day = tmp[tmp["date"] == prev_date][["time_index", "area", "num_people"]].copy()
        prev_day = prev_day.rename(columns={"num_people": "prev_day_same_slot"})
        base = base.merge(prev_day, on=["time_index", "area"], how="left")
    else:
        base["prev_day_same_slot"] = np.nan

    base["slot_mean"] = base["slot_mean"].fillna(0.0)
    base["weekend_slot_mean"] = base["weekend_slot_mean"].fillna(base["slot_mean"])
    base["dow_slot_mean"] = base["dow_slot_mean"].fillna(base["weekend_slot_mean"])
    base["prev_day_same_slot"] = base["prev_day_same_slot"].fillna(base["slot_mean"])

    # 안정성 강화: baseline 비중 상향
    base["baseline_people"] = (
        0.50 * base["slot_mean"] +
        0.20 * base["dow_slot_mean"] +
        0.30 * base["prev_day_same_slot"]
    )

    base["baseline_people"] = base["baseline_people"].clip(lower=0)
    return base[["date", "time_index", "area", "baseline_people"]]

# =========================================================
# 특징 생성
# =========================================================
def make_temporal_features(timestamp_index):
    ts = pd.to_datetime(pd.Series(timestamp_index))

    seconds_in_day = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    frac_day = seconds_in_day / 86400.0
    dow = ts.dt.dayofweek / 7.0
    is_weekend = (ts.dt.dayofweek >= 5).astype(int)

    feats = np.stack([
        np.sin(2 * np.pi * frac_day),
        np.cos(2 * np.pi * frac_day),
        np.sin(2 * np.pi * dow),
        np.cos(2 * np.pi * dow),
        is_weekend.astype(np.float32)
    ], axis=1)

    return feats.astype(np.float32)

def build_log_residual_matrix(pivot_counts: pd.DataFrame, pivot_baseline: pd.DataFrame):
    log_y = np.log1p(pivot_counts.values.astype(np.float32))
    log_b = np.log1p(pivot_baseline.values.astype(np.float32))
    residual = log_y - log_b
    return log_y, log_b, residual

def build_sequence_data(
    count_matrix_log: np.ndarray,
    baseline_matrix_log: np.ndarray,
    residual_matrix: np.ndarray,
    temporal_features: np.ndarray,
    seq_len: int
):
    X, y = [], []

    input_matrix = np.concatenate([
        count_matrix_log,
        baseline_matrix_log,
        residual_matrix,
        temporal_features
    ], axis=1)

    for i in range(seq_len, len(input_matrix)):
        X.append(input_matrix[i-seq_len:i])
        y.append(residual_matrix[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# =========================================================
# 모델
# =========================================================
def build_lstm_model(seq_len: int, feature_dim: int, output_dim: int, lstm_units: int, dropout: float):
    inp = layers.Input(shape=(seq_len, feature_dim))

    x = layers.LSTM(lstm_units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(max(32, lstm_units // 2), return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(output_dim, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model

def make_tf_dataset(X, y, batch_size: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def areawise_best_blend_alpha(actual: np.ndarray, baseline: np.ndarray, model_pred: np.ndarray):
    """
    alpha 범위를 보수적으로 제한해서 과적합/과도한 튐 방지
    """
    alphas = np.linspace(0.0, 0.7, 15)
    n_areas = actual.shape[1]
    best = np.zeros(n_areas, dtype=np.float32)

    for j in range(n_areas):
        best_mae = float("inf")
        best_alpha = 0.3

        a = actual[:, j]
        b = baseline[:, j]
        m = model_pred[:, j]

        for alpha in alphas:
            pred = alpha * m + (1 - alpha) * b
            mae = np.mean(np.abs(a - pred))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        best[j] = best_alpha

    return best

def ema_smooth_per_area(arr: np.ndarray, alpha: float = 0.35):
    out = arr.copy().astype(np.float32)
    for j in range(out.shape[1]):
        for i in range(1, out.shape[0]):
            out[i, j] = alpha * out[i, j] + (1 - alpha) * out[i - 1, j]
    return out

def forecast_with_scaler(
    model,
    y_scaler,
    history_log_counts,
    history_log_baseline,
    history_residual,
    history_temporal,
    future_log_baseline,
    future_temporal,
    seq_len
):
    current_window = np.concatenate([
        history_log_counts[-seq_len:],
        history_log_baseline[-seq_len:],
        history_residual[-seq_len:],
        history_temporal[-seq_len:]
    ], axis=1).astype(np.float32)

    pred_counts = []
    pred_residuals = []
    pred_logs = []

    for i in range(len(future_temporal)):
        pred_scaled = model.predict(current_window[None, :, :], verbose=0)[0]
        pred_res = y_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        pred_log = future_log_baseline[i] + pred_res
        pred_cnt = np.expm1(pred_log)
        pred_cnt = np.clip(pred_cnt, 0, None)

        pred_counts.append(pred_cnt)
        pred_residuals.append(pred_res)
        pred_logs.append(pred_log)

        next_row = np.concatenate([
            pred_log.reshape(1, -1),
            future_log_baseline[i].reshape(1, -1),
            pred_res.reshape(1, -1),
            future_temporal[i].reshape(1, -1)
        ], axis=1)[0]

        current_window = np.vstack([current_window[1:], next_row])

    return (
        np.array(pred_counts, dtype=np.float32),
        np.array(pred_logs, dtype=np.float32),
        np.array(pred_residuals, dtype=np.float32)
    )

# =========================================================
# 성능 유틸
# =========================================================
def calc_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# =========================================================
# 메인
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--file_glob", type=str, default="*.csv")
    parser.add_argument("--output_dir", type=str, default="forecast_output_v3")
    parser.add_argument("--target_date", type=str, required=True)

    parser.add_argument("--raw_interval_seconds", type=int, default=10)
    parser.add_argument("--model_interval_seconds", type=int, default=60)
    parser.add_argument("--seq_len", type=int, default=180, help="60초 기준 180 = 3시간")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lstm_units", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ema_alpha", type=float, default=0.35)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_interval_seconds not in [30, 60, 120, 300]:
        raise ValueError("model_interval_seconds는 30, 60, 120, 300 중 하나를 권장합니다.")

    print("[INFO] 데이터 로딩")
    raw_df, all_dates, all_areas = load_daily_files(
        folder_path=args.data_dir,
        file_glob=args.file_glob,
        raw_interval_seconds=args.raw_interval_seconds
    )

    target_date = pd.to_datetime(args.target_date).date()
    if target_date not in all_dates:
        raise ValueError(f"target_date가 데이터에 없습니다: {target_date}")

    target_idx = all_dates.index(target_date)
    if target_idx < 13:
        raise ValueError("타겟 날짜 이전 13일이 필요합니다.")

    train_dates = all_dates[target_idx - 13: target_idx]
    calib_date = train_dates[-1]
    fit_dates = train_dates[:-1]

    print(f"[INFO] fit_dates  : {fit_dates[0]} ~ {fit_dates[-1]}")
    print(f"[INFO] calib_date : {calib_date}")
    print(f"[INFO] target_date: {target_date}")

    model_df = aggregate_interval(
        raw_df,
        raw_interval_seconds=args.raw_interval_seconds,
        model_interval_seconds=args.model_interval_seconds
    )

    steps_per_day = 86400 // args.model_interval_seconds
    all_areas = sorted(model_df["area"].unique().tolist())

    # -----------------------------------------------------
    # 데이터 분리
    # -----------------------------------------------------
    fit_df = model_df[model_df["date"].isin(fit_dates)].copy()
    calib_df = model_df[model_df["date"] == calib_date].copy()
    target_actual_df = model_df[model_df["date"] == target_date].copy()

    if fit_df.empty or calib_df.empty or target_actual_df.empty:
        raise ValueError("fit/calibration/target 데이터 중 비어 있는 구간이 있습니다.")

    # -----------------------------------------------------
    # target / calib baseline
    # -----------------------------------------------------
    calib_base = build_baseline_for_date(fit_df, calib_date, all_areas, steps_per_day)
    target_base = build_baseline_for_date(
        model_df[model_df["date"].isin(train_dates)].copy(),
        target_date,
        all_areas,
        steps_per_day
    )

    calib_base["actual_time"] = pd.to_datetime(calib_base["date"]) + pd.to_timedelta(
        calib_base["time_index"] * args.model_interval_seconds, unit="s"
    )
    target_base["actual_time"] = pd.to_datetime(target_base["date"]) + pd.to_timedelta(
        target_base["time_index"] * args.model_interval_seconds, unit="s"
    )

    # -----------------------------------------------------
    # fit wide
    # -----------------------------------------------------
    fit_wide = (
        fit_df.pivot_table(index="actual_time", columns="area", values="num_people", aggfunc="sum")
              .sort_index()
              .reindex(columns=all_areas)
              .fillna(0)
    )

    fit_base_long = []
    for d in fit_dates:
        prior_df = fit_df[fit_df["date"] < d].copy()
        if prior_df.empty:
            prior_df = fit_df.copy()

        b = build_baseline_for_date(prior_df, d, all_areas, steps_per_day)
        b["actual_time"] = pd.to_datetime(b["date"]) + pd.to_timedelta(
            b["time_index"] * args.model_interval_seconds, unit="s"
        )
        fit_base_long.append(b)

    fit_base_df = pd.concat(fit_base_long, ignore_index=True)
    fit_base_wide = (
        fit_base_df.pivot_table(index="actual_time", columns="area", values="baseline_people", aggfunc="sum")
                   .sort_index()
                   .reindex(columns=all_areas)
                   .fillna(0)
    )

    fit_base_wide = fit_base_wide.reindex(fit_wide.index).ffill().fillna(0)

    fit_temporal = make_temporal_features(fit_wide.index)
    fit_log_y, fit_log_b, fit_residual = build_log_residual_matrix(fit_wide, fit_base_wide)

    X_train, y_train = build_sequence_data(
        count_matrix_log=fit_log_y,
        baseline_matrix_log=fit_log_b,
        residual_matrix=fit_residual,
        temporal_features=fit_temporal,
        seq_len=args.seq_len
    )

    if len(X_train) == 0:
        raise ValueError("학습 시퀀스를 만들 수 없습니다. seq_len을 줄이세요.")

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    ds_train = make_tf_dataset(X_train, y_train_scaled, batch_size=args.batch_size, shuffle=True)

    # -----------------------------------------------------
    # 모델 학습
    # -----------------------------------------------------
    print(f"[INFO] X_train shape = {X_train.shape}")
    print(f"[INFO] y_train shape = {y_train_scaled.shape}")

    model = build_lstm_model(
        seq_len=args.seq_len,
        feature_dim=X_train.shape[2],
        output_dim=y_train_scaled.shape[1],
        lstm_units=args.lstm_units,
        dropout=args.dropout
    )

    cbs = [
        callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    history = model.fit(
        ds_train,
        epochs=args.epochs,
        verbose=1,
        callbacks=cbs
    )

    pd.DataFrame(history.history).to_csv(
        output_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # calibration 예측
    # -----------------------------------------------------
    print("[INFO] calibration 예측 및 구역별 blend alpha 계산")

    calib_hist_df = model_df[model_df["date"].isin(fit_dates)].copy()

    calib_hist_wide = (
        calib_hist_df.pivot_table(index="actual_time", columns="area", values="num_people", aggfunc="sum")
                    .sort_index()
                    .reindex(columns=all_areas)
                    .fillna(0)
    )

    calib_hist_base_long = []
    for d in fit_dates:
        prior_df = fit_df[fit_df["date"] < d].copy()
        if prior_df.empty:
            prior_df = fit_df.copy()

        b = build_baseline_for_date(prior_df, d, all_areas, steps_per_day)
        b["actual_time"] = pd.to_datetime(b["date"]) + pd.to_timedelta(
            b["time_index"] * args.model_interval_seconds, unit="s"
        )
        calib_hist_base_long.append(b)

    calib_hist_base_df = pd.concat(calib_hist_base_long, ignore_index=True)
    calib_hist_base_wide = (
        calib_hist_base_df.pivot_table(index="actual_time", columns="area", values="baseline_people", aggfunc="sum")
                         .sort_index()
                         .reindex(columns=all_areas)
                         .fillna(0)
    )
    calib_hist_base_wide = calib_hist_base_wide.reindex(calib_hist_wide.index).ffill().fillna(0)

    calib_future_base_wide = (
        calib_base.pivot_table(index="actual_time", columns="area", values="baseline_people", aggfunc="sum")
                 .sort_index()
                 .reindex(columns=all_areas)
                 .fillna(0)
    )

    calib_actual_wide = (
        calib_df.pivot_table(index="actual_time", columns="area", values="num_people", aggfunc="sum")
                .sort_index()
                .reindex(columns=all_areas)
                .fillna(0)
    )

    hist_log_y = np.log1p(calib_hist_wide.values.astype(np.float32))
    hist_log_b = np.log1p(calib_hist_base_wide.values.astype(np.float32))
    hist_res = hist_log_y - hist_log_b
    hist_temporal = make_temporal_features(calib_hist_wide.index)

    fut_log_b = np.log1p(calib_future_base_wide.values.astype(np.float32))
    fut_temporal = make_temporal_features(calib_future_base_wide.index)

    calib_pred_counts_raw, _, _ = forecast_with_scaler(
        model=model,
        y_scaler=y_scaler,
        history_log_counts=hist_log_y,
        history_log_baseline=hist_log_b,
        history_residual=hist_res,
        history_temporal=hist_temporal,
        future_log_baseline=fut_log_b,
        future_temporal=fut_temporal,
        seq_len=args.seq_len
    )

    calib_pred_counts_raw = ema_smooth_per_area(calib_pred_counts_raw, alpha=args.ema_alpha)

    calib_baseline_counts = calib_future_base_wide.values.astype(np.float32)
    calib_actual_counts = calib_actual_wide.values.astype(np.float32)

    blend_alpha = areawise_best_blend_alpha(
        actual=calib_actual_counts,
        baseline=calib_baseline_counts,
        model_pred=calib_pred_counts_raw
    )

    # -----------------------------------------------------
    # 최종 target 예측
    # -----------------------------------------------------
    print("[INFO] target_date 최종 예측")

    final_hist_df = model_df[model_df["date"].isin(train_dates)].copy()

    final_hist_wide = (
        final_hist_df.pivot_table(index="actual_time", columns="area", values="num_people", aggfunc="sum")
                    .sort_index()
                    .reindex(columns=all_areas)
                    .fillna(0)
    )

    final_hist_base_long = []
    for d in train_dates:
        prior_df = model_df[model_df["date"] < d].copy()
        if prior_df.empty:
            prior_df = model_df[model_df["date"].isin(train_dates)].copy()

        b = build_baseline_for_date(prior_df, d, all_areas, steps_per_day)
        b["actual_time"] = pd.to_datetime(b["date"]) + pd.to_timedelta(
            b["time_index"] * args.model_interval_seconds, unit="s"
        )
        final_hist_base_long.append(b)

    final_hist_base_df = pd.concat(final_hist_base_long, ignore_index=True)
    final_hist_base_wide = (
        final_hist_base_df.pivot_table(index="actual_time", columns="area", values="baseline_people", aggfunc="sum")
                         .sort_index()
                         .reindex(columns=all_areas)
                         .fillna(0)
    )
    final_hist_base_wide = final_hist_base_wide.reindex(final_hist_wide.index).ffill().fillna(0)

    target_base_wide = (
        target_base.pivot_table(index="actual_time", columns="area", values="baseline_people", aggfunc="sum")
                  .sort_index()
                  .reindex(columns=all_areas)
                  .fillna(0)
    )

    target_actual_wide = (
        target_actual_df.pivot_table(index="actual_time", columns="area", values="num_people", aggfunc="sum")
                      .sort_index()
                      .reindex(columns=all_areas)
                      .fillna(0)
    )

    hist_log_y = np.log1p(final_hist_wide.values.astype(np.float32))
    hist_log_b = np.log1p(final_hist_base_wide.values.astype(np.float32))
    hist_res = hist_log_y - hist_log_b
    hist_temporal = make_temporal_features(final_hist_wide.index)

    fut_log_b = np.log1p(target_base_wide.values.astype(np.float32))
    fut_temporal = make_temporal_features(target_base_wide.index)

    pred_counts_raw, _, _ = forecast_with_scaler(
        model=model,
        y_scaler=y_scaler,
        history_log_counts=hist_log_y,
        history_log_baseline=hist_log_b,
        history_residual=hist_res,
        history_temporal=hist_temporal,
        future_log_baseline=fut_log_b,
        future_temporal=fut_temporal,
        seq_len=args.seq_len
    )

    pred_counts_raw = ema_smooth_per_area(pred_counts_raw, alpha=args.ema_alpha)

    baseline_counts = target_base_wide.values.astype(np.float32)

    pred_counts_blend = (
        blend_alpha.reshape(1, -1) * pred_counts_raw +
        (1 - blend_alpha.reshape(1, -1)) * baseline_counts
    )

    # 상한/하한 보정
    caps_hi = (
        final_hist_df.groupby("area")["num_people"]
                    .quantile(0.995)
                    .reindex(all_areas)
                    .fillna(final_hist_df["num_people"].max())
                    .values.astype(np.float32)
    )
    caps_lo = (
        final_hist_df.groupby("area")["num_people"]
                    .quantile(0.05)
                    .reindex(all_areas)
                    .fillna(0)
                    .values.astype(np.float32)
    )

    caps_hi = np.maximum(caps_hi, 1.0)
    caps_lo = np.maximum(caps_lo, 0.0)

    pred_counts_final = np.minimum(pred_counts_blend, caps_hi.reshape(1, -1) * 1.20)
    pred_counts_final = np.maximum(pred_counts_final, np.minimum(caps_lo.reshape(1, -1) * 0.50, pred_counts_final))
    pred_counts_final = np.clip(pred_counts_final, 0, None)
    pred_counts_final = np.round(pred_counts_final).astype(int)

    # -----------------------------------------------------
    # long 변환
    # -----------------------------------------------------
    pred_wide_df = pd.DataFrame(pred_counts_final, index=target_actual_wide.index, columns=all_areas)
    baseline_wide_df = pd.DataFrame(baseline_counts, index=target_actual_wide.index, columns=all_areas)

    pred_long = pred_wide_df.reset_index().melt(
        id_vars="actual_time", var_name="area", value_name="num_people"
    )
    pred_long["date"] = target_date
    pred_long["time_index"] = (
        (pred_long["actual_time"] - pd.to_datetime(target_date)).dt.total_seconds() // args.model_interval_seconds
    ).astype(int)
    pred_long["kind"] = "pred"

    actual_long = target_actual_wide.reset_index().melt(
        id_vars="actual_time", var_name="area", value_name="num_people"
    )
    actual_long["date"] = target_date
    actual_long["time_index"] = (
        (actual_long["actual_time"] - pd.to_datetime(target_date)).dt.total_seconds() // args.model_interval_seconds
    ).astype(int)
    actual_long["kind"] = "actual"

    baseline_long = baseline_wide_df.reset_index().melt(
        id_vars="actual_time", var_name="area", value_name="num_people"
    )
    baseline_long["date"] = target_date
    baseline_long["time_index"] = (
        (baseline_long["actual_time"] - pd.to_datetime(target_date)).dt.total_seconds() // args.model_interval_seconds
    ).astype(int)
    baseline_long["kind"] = "baseline"
    baseline_long["num_people"] = baseline_long["num_people"].round().astype(int)

    pred_long = add_wait_time_columns(pred_long, interval_seconds=args.model_interval_seconds)
    actual_long = add_wait_time_columns(actual_long, interval_seconds=args.model_interval_seconds)
    baseline_long = add_wait_time_columns(baseline_long, interval_seconds=args.model_interval_seconds)

    compare_df = pred_long.merge(
        actual_long[["actual_time", "area", "num_people", "wait_time_min"]],
        on=["actual_time", "area"],
        how="left",
        suffixes=("_pred", "_actual")
    )

    compare_df = compare_df.merge(
        baseline_long[["actual_time", "area", "num_people", "wait_time_min"]],
        on=["actual_time", "area"],
        how="left"
    ).rename(columns={
        "num_people": "num_people_baseline",
        "wait_time_min": "wait_time_min_baseline"
    })

    compare_df["abs_err_people"] = (compare_df["num_people_pred"] - compare_df["num_people_actual"]).abs()
    compare_df["abs_err_wait"] = (compare_df["wait_time_min_pred"] - compare_df["wait_time_min_actual"]).abs()
    compare_df["abs_err_people_baseline"] = (compare_df["num_people_baseline"] - compare_df["num_people_actual"]).abs()
    compare_df["abs_err_wait_baseline"] = (compare_df["wait_time_min_baseline"] - compare_df["wait_time_min_actual"]).abs()

    # -----------------------------------------------------
    # 성능표
    # -----------------------------------------------------
    overall_metrics_df = pd.DataFrame([{
        "target_date": str(target_date),
        "train_start_date": str(train_dates[0]),
        "train_end_date": str(train_dates[-1]),
        "fit_end_date": str(fit_dates[-1]),
        "calib_date": str(calib_date),
        "num_areas": len(all_areas),
        "raw_interval_seconds": args.raw_interval_seconds,
        "model_interval_seconds": args.model_interval_seconds,
        "seq_len": args.seq_len,
        "people_MAE_pred": mean_absolute_error(compare_df["num_people_actual"], compare_df["num_people_pred"]),
        "people_RMSE_pred": calc_rmse(compare_df["num_people_actual"], compare_df["num_people_pred"]),
        "people_MAE_baseline": mean_absolute_error(compare_df["num_people_actual"], compare_df["num_people_baseline"]),
        "people_RMSE_baseline": calc_rmse(compare_df["num_people_actual"], compare_df["num_people_baseline"]),
        "wait_MAE_pred": mean_absolute_error(compare_df["wait_time_min_actual"], compare_df["wait_time_min_pred"]),
        "wait_RMSE_pred": calc_rmse(compare_df["wait_time_min_actual"], compare_df["wait_time_min_pred"]),
        "wait_MAE_baseline": mean_absolute_error(compare_df["wait_time_min_actual"], compare_df["wait_time_min_baseline"]),
        "wait_RMSE_baseline": calc_rmse(compare_df["wait_time_min_actual"], compare_df["wait_time_min_baseline"]),
    }])

    area_rows = []
    for i, area in enumerate(all_areas):
        sub = compare_df[compare_df["area"] == area].copy()

        people_mae_pred = mean_absolute_error(sub["num_people_actual"], sub["num_people_pred"])
        people_mae_base = mean_absolute_error(sub["num_people_actual"], sub["num_people_baseline"])
        wait_mae_pred = mean_absolute_error(sub["wait_time_min_actual"], sub["wait_time_min_pred"])
        wait_mae_base = mean_absolute_error(sub["wait_time_min_actual"], sub["wait_time_min_baseline"])

        denom = sub["num_people_actual"].replace(0, np.nan)
        mape_pred = ((sub["num_people_pred"] - sub["num_people_actual"]).abs() / denom).mean() * 100
        mape_base = ((sub["num_people_baseline"] - sub["num_people_actual"]).abs() / denom).mean() * 100

        area_rows.append({
            "area": area,
            "area_type": classify_area(area),
            "blend_alpha": round(float(blend_alpha[i]), 3),
            "people_MAE_pred": round(float(people_mae_pred), 3),
            "people_MAE_baseline": round(float(people_mae_base), 3),
            "people_MAPE_pred(%)": round(float(mape_pred), 3) if not pd.isna(mape_pred) else np.nan,
            "people_MAPE_baseline(%)": round(float(mape_base), 3) if not pd.isna(mape_base) else np.nan,
            "wait_MAE_pred": round(float(wait_mae_pred), 3),
            "wait_MAE_baseline": round(float(wait_mae_base), 3),
            "improved_vs_baseline_people": bool(people_mae_pred < people_mae_base),
            "improved_vs_baseline_wait": bool(wait_mae_pred < wait_mae_base),
        })

    area_metrics_df = pd.DataFrame(area_rows).sort_values(["area_type", "area"])

    blend_df = pd.DataFrame({
        "area": all_areas,
        "blend_alpha": blend_alpha
    })

    # -----------------------------------------------------
    # 저장
    # -----------------------------------------------------
    pred_long.to_csv(output_dir / "pred_people_wait.csv", index=False, encoding="utf-8-sig")
    actual_long.to_csv(output_dir / "actual_people_wait.csv", index=False, encoding="utf-8-sig")
    baseline_long.to_csv(output_dir / "baseline_people_wait.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(output_dir / "compare_pred_vs_actual.csv", index=False, encoding="utf-8-sig")
    overall_metrics_df.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    area_metrics_df.to_csv(output_dir / "area_metrics.csv", index=False, encoding="utf-8-sig")
    blend_df.to_csv(output_dir / "blend_alpha.csv", index=False, encoding="utf-8-sig")

    model.save(output_dir / "lstm_model_v3.keras")

    print("[INFO] 저장 완료")
    print(f"[INFO] output_dir = {output_dir.resolve()}")
    print("[INFO] 생성 파일:")
    for name in [
        "training_history.csv",
        "pred_people_wait.csv",
        "actual_people_wait.csv",
        "baseline_people_wait.csv",
        "compare_pred_vs_actual.csv",
        "overall_metrics.csv",
        "area_metrics.csv",
        "blend_alpha.csv",
        "lstm_model_v3.keras"
    ]:
        print(f" - {name}")

if __name__ == "__main__":
    main()
