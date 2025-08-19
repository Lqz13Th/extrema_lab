import polars as pl
from polars import Expr
from typing import List

EPSILON = 1e-8


def rolling_z_tanh_normalize(
    rollin_df: pl.DataFrame,
    window: int,
    rolling_mean_window: int,
) -> pl.DataFrame:
    columns_to_normalize = [
        col for col in rollin_df.columns
        if col not in ['px', 'timestamp', 'timestamp_dt', 'symbol']
           and not col.startswith("future_return")
           and not col.endswith('scaled')
    ]

    return rollin_df.with_columns([
        z_score_tanh_expr(
            col=column,
            scaled_col=f"{column}_zscaled",
            window=window,
            rolling_mean_window=rolling_mean_window,
        ) for column in columns_to_normalize
    ])

def z_score_tanh_expr(
    col: str,
    scaled_col: str,
    window: int,
    rolling_mean_window: int,
) -> Expr:
    mean_expr = pl.col(col).rolling_mean(window, min_samples=1)
    std_expr = pl.col(col).rolling_std(window, min_samples=1).fill_nan(0)

    return (
        ((pl.col(col) - mean_expr) / (std_expr + EPSILON))
        .fill_nan(0)
        .fill_null(0)
        .clip(-3.0, 3.0)
        .tanh()
        .rolling_mean(rolling_mean_window, min_samples=1)
        .alias(scaled_col)
    )

def z_score_expr(
    col: str,
    window: int
) -> Expr:
    mean_expr = pl.col(col).rolling_mean(window, min_samples=1)
    std_expr = pl.col(col).rolling_std(window, min_samples=1).fill_nan(0)

    return (
        ((pl.col(col) - mean_expr) / (std_expr + EPSILON))
        .fill_nan(0)
        .fill_null(0)
        .clip(-3.0, 3.0)
        .alias(f"z_{col}")
    )

def pct_change_expr(col: str) -> Expr:
    return pl.col(col).pct_change().fill_nan(0).alias(f"{col}_pct_change")

def lag_exprs(col: str, lag: int) -> Expr:
    return pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")

def diff_expr(col: str, lag: int = 1) -> Expr:
    return pl.col(col).diff(lag).alias(f"{col}_diff_{lag}")

def rolling_std_expr(col: str, window: int) -> Expr:
    return pl.col(col).rolling_std(window, min_samples=1).alias(f"{col}_std_{window}")

def rolling_sum_expr(col: str, window: int) -> Expr:
    return pl.col(col).rolling_sum(window, min_samples=1).alias(f"{col}_sum_{window}")


def rolling_pct_change_std_expr(col: str, window: int) -> Expr:
    return (
        pl.col(col)
        .pct_change()
        .fill_nan(0)
        .rolling_std(window, min_samples=1)
        .alias(f"{col}_pct_change_std_{window}")
    )

def rolling_pct_change_sum_expr(col: str, window: int) -> pl.Expr:
    return (
        pl.col(col)
        .pct_change()
        .fill_nan(0)
        .rolling_sum(window, min_samples=1)
        .alias(f"{col}_pct_change_sum_{window}")
    )

def rolling_mean_ratio_expr(col: str, window: int) -> Expr:
    return (
        (pl.col(col) / pl.col(col).rolling_mean(window, min_samples=1))
        .alias(f"{col}_roll_mean_ratio_{window}")
    )

def rolling_ma_ratio_expr(col: str, short: int, long: int) -> Expr:
    short_ma = pl.col(col).rolling_mean(short, min_samples=1)
    long_ma = pl.col(col).rolling_mean(long, min_samples=1)
    return (short_ma / (long_ma + EPSILON)).alias(f"{col}_ma_ratio_{short}_{long}")

def rolling_pct_change_std_ratio_expr(col: str, short: int = 12, long: int = 72) -> Expr:
    short_vol = pl.col(col).pct_change().rolling_std(short, min_samples=1)
    long_vol = pl.col(col).pct_change().rolling_std(long, min_samples=1)
    return (short_vol / (long_vol + EPSILON)).alias(f"{col}_pct_change_std_ratio")

def second_order_diff_expr(col: str, lag: int = 1) -> Expr:
    first_diff = pl.col(col) - pl.col(col).shift(lag)
    second_diff = first_diff - first_diff.shift(lag)
    return second_diff.alias(f"{col}_second_order_diff_{lag}")

def momentum_ratio_expr(col: str, window: int = 200) -> Expr:
    return (
        ((pl.col(col).abs() + EPSILON).log1p() - (pl.col(col).abs() + EPSILON).shift(window).log1p())
        .alias(f"{col}_momentum_ratio_{window}")
    )

def cross_minus_expr(a: str, b: str) -> Expr:
    return (pl.col(a) - (pl.col(b) + EPSILON)).alias(f"{a}_minus_{b}")

def rsi_expr(col: str, window: int) -> Expr:
    delta = pl.col(col).diff().fill_null(0)
    up = delta.clip(lower_bound=0)
    down = (-delta).clip(lower_bound=0)
    avg_up = up.rolling_mean(window, min_samples=1)
    avg_down = down.rolling_mean(window, min_samples=1)
    rsi = 100 - 100 / (1 + avg_up / (avg_down + 1e-6))
    return rsi.alias(f"{col}_rsi_{window}")

def atr_expr(col: str, high: str, low: str, close: str, window: int) -> Expr:
    tr = pl.max_horizontal([
        pl.col(high) - pl.col(low),
        (pl.col(high) - pl.col(close).shift(1)).abs(),
        (pl.col(low) - pl.col(close).shift(1)).abs()
    ])
    return tr.rolling_mean(window, min_samples=1).alias(f"{col}_atr_{window}")

def cols_to_transforms(
        df: pl.DataFrame,
        exclude_cols: List[str] = None
) -> List[str]:
    if exclude_cols is None:
        exclude_cols = ['px', 'timestamp', 'timestamp_dt', 'symbol']

    if isinstance(df, pl.LazyFrame):
        cols = df.collect_schema().names()
    else:
        cols = df.columns

    cols = [
        col for col in cols
        if col not in exclude_cols and not (
                col.endswith('_rolling_mean') or
                col.endswith('_rolling_std') or
                col.endswith('_scaled')
        ) and not col.startswith("z_")
    ]

    return cols

def batch_apply_single_exprs(
        window: int,
        lags,
        cols: List[str] = None
) -> List[str]:
    single_exprs = []
    # single features transformation
    for col in cols:
        single_exprs.extend([
            momentum_ratio_expr(col, window),
            rolling_std_expr(col, window),
        ])
        for lag in lags:
            single_exprs.extend([
                lag_exprs(col, lag),
                diff_expr(col, lag),
                second_order_diff_expr(col, lag),
            ])

    return single_exprs

def batch_apply_multi_exprs(
        cols: List[str] = None
) -> List[str]:
    multi_exprs = []

    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            multi_exprs.extend([
                cross_minus_expr(a, b),
            ])

    return multi_exprs

def batch_apply_transforms(
        df_to_transforms: pl.DataFrame,
        window: int,
        lags,
        log1p_cols: List[str] = None,
        exclude_cols: List[str] = None,
) -> pl.DataFrame:
    if exclude_cols is None:
        exclude_cols = ['px', 'timestamp', 'timestamp_dt', 'symbol']

    if log1p_cols is None:
        log1p_cols = []

    for col in log1p_cols:
        if col in df_to_transforms.columns:
            df_to_transforms = df_to_transforms.with_columns([
                pl.col(col).clip(lower_bound=0.0).log1p().alias(col)
            ])

    base_cols = cols_to_transforms(df_to_transforms, exclude_cols)

    single_exprs = batch_apply_single_exprs(window, lags, base_cols)
    multi_exprs = batch_apply_multi_exprs(base_cols)

    exprs = single_exprs + multi_exprs
    return df_to_transforms.with_columns(exprs)

def convert_all_to_float64_except_timestamp(df_cast: pl.DataFrame) -> pl.DataFrame:
    float_cols = [c for c in df_cast.columns if c != "timestamp"]
    return df_cast.with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])

def rename_with_prefix(df_prefix: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return df_prefix.rename({
        col_prefix: f"{prefix}{col_prefix}" for col_prefix in df_prefix.columns if col_prefix != "timestamp"
    })

def merge_dataframes_on_timestamp(dfs: list[pl.DataFrame], prefixes: list[str]) -> pl.DataFrame:
    assert len(dfs) == len(prefixes)

    dfs_renamed = [rename_with_prefix(df, prefix) for df, prefix in zip(dfs, prefixes)]
    merged = dfs_renamed[0]

    for df_rename in dfs_renamed[1:]:
        merged = (
            merged
            .join(df_rename, on="timestamp", how="full")
            .with_columns([
                pl.coalesce([pl.col("timestamp"), pl.col("timestamp_right")]).alias("timestamp")
            ])
            .drop("timestamp_right")
        )

    return merged.sort("timestamp")

def auto_fill_dataframes_with_old_data(auto_fill_df: pl.DataFrame) -> pl.DataFrame:
    auto_fill_df = auto_fill_df.sort("timestamp")
    columns_to_fill = auto_fill_df.columns
    for col_fill in columns_to_fill:
        auto_fill_df = auto_fill_df.with_columns(
            pl
            .col(col_fill)
            .fill_null(strategy="forward")
            .alias(col_fill)
        )

    return auto_fill_df

