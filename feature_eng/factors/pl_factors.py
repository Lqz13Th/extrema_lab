from feature_eng.operator.pl_operators import *

EPSILON = 1e-8

# alt merged data
def premium_oi_dev_expr(window: int = 12) -> pl.Expr:
    premium_ma = pl.col("premium_oi").rolling_mean(window, min_samples=1)
    return ((pl.col("premium_oi") - premium_ma) / (premium_ma + EPSILON)).alias(f"premium_oi_dev_{window}")

def alt_basic_transform() -> List[Expr]:
    return [
        (pl.col("adjusted_funding_rate") * pl.col("oi_sum_open_interest"))
        .alias("funding_oi"),

        (pl.col("oi_sum_open_interest") * pl.col("premium_open"))
        .alias("premium_oi"),

        (pl.col("adjusted_funding_rate") * pl.col("premium_open"))
        .alias("funding_premium"),

        (pl.col("adjusted_funding_rate") *
         pl.col("oi_sum_open_interest") *
         pl.col("premium_open"))
        .alias("factor_triplet"),

        (pl.col("premium_open") - pl.col("adjusted_funding_rate"))
        .alias("premium_funding_spread"),
    ]

def alt_basic_factors(windows: List[int]) -> List[Expr]:
    single_exprs = []
    for window in windows:
        single_exprs.extend([
            rolling_mean_ratio_expr("oi_sum_open_interest", window=window),
            rolling_pct_change_sum_expr("oi_sum_open_interest", window=window),
        ])

    return single_exprs

def alt_basic_non_sensitive_factors(window: int) -> List[Expr]:
    return [
        rolling_sum_expr("premium_funding_spread", window=window),
        rolling_sum_expr("adjusted_funding_rate", window=window),
        rolling_sum_expr("premium_open", window=window),
        rolling_mean_ratio_expr("funding_oi", window=window),
        rolling_mean_ratio_expr("premium_oi", window=window),
        rolling_mean_ratio_expr("factor_triplet", window=window),
    ]

def alt_advanced_factors(windows: int) -> List[Expr]:
    advanced_exprs = [

        premium_oi_dev_expr(windows),

    ]
    return advanced_exprs

def factor_px_pct_rolling(df: pl.DataFrame, window: int):
    for w in [10, 20, 40, 80, 160, window]:
        df = df.with_columns([
            pl.col("px_pct").rolling_sum(w, min_samples=1).alias(f"px_pct_rol_sum_{w}"),
            z_score_expr(f"px_pct_rol_sum_{w}", window)
        ])
    return df

def factor_lob(df: pl.DataFrame, window: int):
    df = df.with_columns([
        ((pl.col("far_bid_price") - pl.col("best_bid_price")).abs().rolling_mean(window, min_samples=1))
        .alias(f"bid_px_gap_rol_mean_{window}"),
        ((pl.col("far_ask_price") - pl.col("best_ask_price")).abs().rolling_mean(window, min_samples=1))
        .alias(f"ask_px_gap_rol_mean_{window}"),
        ((pl.col("real_bid_amount_sum") - pl.col("real_ask_amount_sum")).rolling_mean(window, min_samples=1))
        .alias(f"lob_ratio_rol_mean_{window}"),
        z_score_expr(f"lob_ratio_rol_mean_{window}", window)
    ])
    return df

def factor_oi(df: pl.DataFrame, window: int):
    df = df.with_columns([
        pl.col("sum_open_interest").clip(lower_bound=0).log1p().alias("log1p_sum_open_interest"),
        z_score_expr("log1p_sum_open_interest", window),
        (pl.col("raw_factor_short_term_oi_trend") - pl.col("raw_factor_long_term_oi_trend"))
        .alias("oi_ls_term_diff"),
        z_score_expr("oi_ls_term_diff", window)
    ])
    return df

def factor_oi_px_divergence(df: pl.DataFrame, window: int):
    # 短期
    px_sign = pl.when(pl.col(f"z_px_pct_rol_sum_{window}") > 0).then(1)\
                 .when(pl.col(f"z_px_pct_rol_sum_{window}") < 0).then(-1).otherwise(0)
    oi_sign = pl.when(pl.col("z_factor_oi_change") > 0).then(1)\
                 .when(pl.col("z_factor_oi_change") < 0).then(-1).otherwise(0)
    is_divergent = (px_sign * oi_sign) < 0
    df = df.with_columns([
        ((pl.col(f"z_px_pct_rol_sum_{window}") + pl.col("z_factor_oi_change").abs()) * is_divergent.cast(pl.Int8))
        .alias("factor_oi_px_divergence_with_sign")
    ])
    return df

