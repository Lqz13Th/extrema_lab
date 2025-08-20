import gc
import os
import numpy as np

from tqdm import tqdm
from ..factors.pl_factors import *

def split_df_by_month(
        origin_input_df: pl.DataFrame,
        ts_col: str = "timestamp"
) -> List[pl.DataFrame]:
    origin_input_df = origin_input_df.with_columns([
        pl.col(ts_col).cast(pl.Datetime("us")).alias(f"{ts_col}_dt")
    ])
    origin_input_df = origin_input_df.with_columns([
        pl.col(f"{ts_col}_dt").dt.truncate("1mo").alias("month_start")
    ])

    unique_months = origin_input_df.select("month_start").unique().sort("month_start")
    monthly_dfs = [
        origin_input_df.filter(pl.col("month_start") == mt).drop("month_start")
        for mt in unique_months["month_start"]
    ]
    return monthly_dfs

def split_df_by_week(
        origin_input_df: pl.DataFrame,
        ts_col: str = "timestamp"
) -> List[pl.DataFrame]:
    origin_input_df = origin_input_df.with_columns([
        pl.col(ts_col).cast(pl.Datetime("us")).alias(f"{ts_col}_dt")
    ])
    origin_input_df = origin_input_df.with_columns([
        pl.col(f"{ts_col}_dt").dt.truncate("1w").alias("week_start")
    ])

    unique_weeks = origin_input_df.select("week_start").unique().sort("week_start")
    weekly_dfs = [
        origin_input_df.filter(pl.col("week_start") == wk).drop("week_start")
        for wk in unique_weeks["week_start"]
    ]
    return weekly_dfs

def clean_df_drop_nulls(
        df_to_clean: pl.DataFrame,
        null_threshold: int = 500000,
        verbose: bool = True
) -> pl.DataFrame:
    pd_df = df_to_clean.to_pandas()
    del df_to_clean
    gc.collect()
    print("converted")

    null_counts = pd_df.isnull().sum()
    cols_to_drop = null_counts[null_counts > null_threshold].index
    pd_df_cleaned = pd_df.drop(columns=cols_to_drop)
    if verbose:
        print("各列空值数量：")
        print(null_counts[null_counts > 0])
        print(f"删除空值超过 {null_threshold} 的列：{list(cols_to_drop)}")
        print(f"删除列后，DataFrame形状：{pd_df_cleaned.shape}")

    del pd_df
    gc.collect()

    pd_df_clean = pd_df_cleaned.dropna()
    if verbose:
        print(f"删除空值行后，DataFrame形状：{pd_df_clean.shape}")

    del pd_df_cleaned
    gc.collect()
    pl_df_clean = pl.from_pandas(pd_df_clean)

    del pd_df_clean
    gc.collect()
    return pl_df_clean

def avg_steps_to_volatility(prices: np.ndarray, target_ratio: float) -> int:
    n = len(prices)
    steps_list = []
    for i in tqdm(range(n), desc=f"cal abs change {target_ratio*100:.2f}% avg steps"):
        start_price = prices[i]
        steps = -1
        for j in range(i + 1, n):
            change = abs(prices[j] / start_price - 1)
            if change >= target_ratio:
                steps = j - i
                break
        if steps != -1:
            steps_list.append(steps)
    if len(steps_list) == 0:
        return -1
    return int(np.mean(steps_list))

def future_return_expr(price_col: str, step: int) -> pl.Expr:
    return ((pl.col(price_col).shift(-step) - pl.col(price_col)) / pl.col(price_col)).alias(f"future_return_{step}")

def filter_valid_cross_sections(long_df: pl.DataFrame, num_symbols: int) -> pl.DataFrame:
    valid_ts = (
        long_df
        .filter(pl.col("px").is_not_null())
        .group_by("timestamp")
        .agg(pl.len().alias("len"))
        .filter(pl.col("len") == num_symbols)
        .select("timestamp")
    )
    return long_df.join(valid_ts, on="timestamp", how="inner")

def build_long_cross_sections_fast(symbol_dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    all_timestamps = (
        pl.concat(
            [df.select("timestamp").unique() for df in symbol_dfs.values()]
        )
        .unique()
        .sort("timestamp")
    )

    result_dfs = []
    for symbol, df in symbol_dfs.items():
        df_sorted = df.sort("timestamp")
        joined = all_timestamps.join_asof(
            df_sorted,
            on="timestamp",
            strategy="backward",
        )
        joined = joined.with_columns(pl.lit(symbol).alias("symbol"))
        result_dfs.append(joined)

    long_df = pl.concat(result_dfs).sort(["timestamp", "symbol"])
    clean_df = filter_valid_cross_sections(long_df, len(symbol_dfs))
    return clean_df

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "extrema_lab"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_proc", "resampled_data") + os.sep

def process_single_symbol(
        symbol: str,
        threshold: str,
        feat_cal_window: int = 500,
        feat_norm_window: int = 500,
        feat_norm_rolling_mean_window: int = 50,
        data_dir: str = DATA_DIR,
) -> pl.DataFrame:
    file = f"{symbol}_merged_thr{threshold}.parquet"
    path = os.path.join(data_dir, file)
    df = pl.read_parquet(path)
    df = cal_factors_with_sampled_data(df, feat_cal_window)
    df = rolling_z_tanh_normalize(df, feat_norm_window, feat_norm_rolling_mean_window)

    stds = df.select([
        pl.col(col).std().alias(col)
        for col in df.columns
        if df[col].dtype in (pl.Float64, pl.Int64)
    ])
    # zero_std_cols = [col for col in stds.columns if stds[0, col] == 0.0]
    # print(zero_std_cols)
    # df = df.drop(zero_std_cols)
    return df

def process_all_symbols(params_dict):
    symbol_dfs = {}
    for sym, param in params_dict.items():
        df = process_single_symbol(
            symbol=sym,
            threshold=param.get("threshold", "0.002"),
            feat_cal_window=param.get("feat_cal_window", 500),
            feat_norm_window=param.get("feat_norm_window", 500),
            feat_norm_rolling_mean_window=param.get("feat_norm_rolling_mean_window", 50),
        )
        df = df.with_columns(pl.lit(sym).alias("symbol"))
        symbol_dfs[sym] = df

    return symbol_dfs

def rolling_ic_ir_icto_index(
        df: pl.DataFrame,
        target_col: str,
        window_size: int,
        step: int = 1,
) -> pl.DataFrame:
    feature_cols = [col for col in df.columns if "scaled" in col]

    n = df.height
    results = []
    prev_ranks = {}

    for start in tqdm(range(0, n - window_size + 1, step), desc="Rolling IC & ICTO"):
        end = start + window_size
        df_win = df.slice(start, window_size)

        df_ranked = df_win.with_columns([
            (pl.col(c).rank(method="average") / window_size).alias(c + "_rank") for c in feature_cols + [target_col]
        ])
        target_rank_col = target_col + "_rank"

        for feat in feature_cols:
            feat_rank_col = feat + "_rank"
            ic = df_ranked.select(
                pl.corr(pl.col(feat_rank_col), pl.col(target_rank_col)).alias("ic")
            ).to_series()[0]

            turnover = None
            if feat in prev_ranks:
                cur_ranks = df_ranked[feat_rank_col].to_numpy()
                prev = prev_ranks[feat]
                if len(prev) == len(cur_ranks):
                    turnover = np.mean(np.abs(cur_ranks - prev))

            prev_ranks[feat] = df_ranked[feat_rank_col].to_numpy()

            results.append({
                "window_start": int(start),
                "window_end": int(end - 1),
                "factor": str(feat),
                "ic": float(ic) if not np.isnan(ic) else None,
                "turnover": float(turnover) if turnover is not None else None
            })

    df_result = pl.DataFrame(
        results,
        schema={
            "window_start": pl.Int64,
            "window_end": pl.Int64,
            "factor": pl.Utf8,
            "ic": pl.Float64,
            "turnover": pl.Float64,
        }
    )
    return (
        df_result
        .group_by("factor")
        .agg([
            pl.mean("ic").alias("mean_ic"),
            pl.std("ic").alias("std_ic"),
            pl.mean("turnover").alias("mean_turnover")
        ])
        .with_columns([
            (pl.col("mean_ic") / pl.col("std_ic")).alias("ir"),
            (pl.col("mean_ic") / (pl.col("mean_turnover") + EPSILON)).abs().alias("icto")
        ])
        .sort("icto", descending=True)
    )