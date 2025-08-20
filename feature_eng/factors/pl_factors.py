from ..operator.pl_operators import *
from polars import functions as F

EPSILON = 1e-8

# alt merged data
def premium_oi_dev_expr(window: int = 12) -> pl.Expr:
    premium_ma = pl.col("premium_oi").rolling_mean(window, min_samples=1)
    return ((pl.col("premium_oi") - premium_ma) / (premium_ma + EPSILON)).alias(f"premium_oi_dev_{window}")

def alt_basic_transform() -> List[Expr]:
    return [
        (pl.col("adjusted_funding_rate") * pl.col("oi_sum_open_interest_value"))
        .alias("funding_oi"),

        (pl.col("oi_sum_open_interest_value") * pl.col("premium_open"))
        .alias("premium_oi"),

        (pl.col("adjusted_funding_rate") * pl.col("premium_open"))
        .alias("funding_premium"),

        (pl.col("adjusted_funding_rate") *
         pl.col("oi_sum_open_interest_value") *
         pl.col("premium_open"))
        .alias("factor_triplet"),

        (pl.col("premium_open") - pl.col("adjusted_funding_rate"))
        .alias("premium_funding_spread"),
    ]

def alt_basic_factors(windows: List[int]) -> List[Expr]:
    single_exprs = []
    for window in windows:
        single_exprs.extend([
            rolling_mean_ratio_expr("oi_sum_open_interest_value", window=window),
            rolling_pct_change_sum_expr("oi_sum_open_interest_value", window=window),
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

# resampled data
def tick_basic_factors(window: int) -> List[Expr]:
    return [
        pl.col("px").rolling_max(window, min_samples=1).alias(f"px_max_{window}"),
        pl.col("px").rolling_min(window, min_samples=1).alias(f"px_min_{window}"),

        pl.col("px_pct").rolling_sum(10, min_samples=1).alias(f"px_pct_rol_sum_{10}"),
        pl.col("px_pct").rolling_sum(20, min_samples=1).alias(f"px_pct_rol_sum_{20}"),
        pl.col("px_pct").rolling_sum(40, min_samples=1).alias(f"px_pct_rol_sum_{40}"),
        pl.col("px_pct").rolling_sum(80, min_samples=1).alias(f"px_pct_rol_sum_{80}"),
        pl.col("px_pct").rolling_sum(160, min_samples=1).alias(f"px_pct_rol_sum_{160}"),
        pl.col("px_pct").rolling_sum(window, min_samples=1).alias(f"px_pct_rol_sum_{window}"),
    ]

def trade_size_basic_factors(window: int) -> List[Expr]:
    return [
        (pl.col("sum_buy_sz") + pl.col("sum_sell_sz"))
        .clip(lower_bound=0.0)
        .log1p()
        .alias(f"log1p_sum_sz_{window}"),

        ((pl.col("sum_buy_sz") - pl.col("sum_sell_sz")) /
         (pl.col("sum_buy_sz") + pl.col("sum_sell_sz") + EPSILON))
        .rolling_mean(window, min_samples=1)
        .alias(f"bs_ratio_rol_mean_{window}"),

        pl.col("bs_imbalance").rolling_mean(window, min_samples=1)
        .alias(f"bs_imba_rol_mean_{window}"),

        pl.col("bs_imbalance").rolling_sum(window, min_samples=1)
        .alias(f"bs_imba_rol_sum_{window}"),
    ]

def oi_div_factors(window: int) -> List[Expr]:
    return [
        pl.when(
            (pl.col(f"px_pct_rol_sum_{window}") > 0) & (pl.col("oi_sum_open_interest_value_pct_change_sum_72") < 0)
        ).then(
            pl.col(f"px_pct_rol_sum_{window}") * pl.col("oi_sum_open_interest_value_pct_change_sum_72").abs()
        ).otherwise(0.0)
        .alias("oi_up_divergence"),

        pl.when(
            (pl.col(f"px_pct_rol_sum_{window}") < 0) & (pl.col("oi_sum_open_interest_value_pct_change_sum_72") > 0)
        ).then(
            pl.col(f"px_pct_rol_sum_{window}") * pl.col("oi_sum_open_interest_value_pct_change_sum_72").abs()
        ).otherwise(0.0)
        .alias("oi_dn_divergence"),

        pl.when(
            (pl.col(f"px_pct_rol_sum_{window}") > 0) & (pl.col("oi_sum_open_interest_value_pct_change_sum_288") < 0)
        ).then(
            pl.col(f"px_pct_rol_sum_{window}") * pl.col("oi_sum_open_interest_value_pct_change_sum_288").abs()
        ).otherwise(0.0)
        .alias("oi_up_divergence_long_term"),

        pl.when(
            (pl.col(f"px_pct_rol_sum_{window}") < 0) & (pl.col("oi_sum_open_interest_value_pct_change_sum_288") > 0)
        ).then(
            pl.col(f"px_pct_rol_sum_{window}") * pl.col("oi_sum_open_interest_value_pct_change_sum_288").abs()
        ).otherwise(0.0)
        .alias("oi_dn_divergence_long_term"),

        pl.when(
            (pl.col(f"px_pct_rol_sum_{window}") > 0) & (pl.col("oi_sum_open_interest_value_pct_change_sum_12") < 0)
        ).then(
            pl.col(f"px_pct_rol_sum_{window}") * pl.col("oi_sum_open_interest_value_pct_change_sum_12").abs()
        ).otherwise(0.0)
        .alias("oi_up_divergence_short_term"),

        pl.when(
            (pl.col(f"px_pct_rol_sum_{window}") < 0) & (pl.col("oi_sum_open_interest_value_pct_change_sum_12") > 0)
        ).then(
            pl.col(f"px_pct_rol_sum_{window}") * pl.col("oi_sum_open_interest_value_pct_change_sum_12").abs()
        ).otherwise(0.0)
        .alias("oi_dn_divergence_short_term"),
    ]

def log_feat_scaling_factors(window: int) -> List[Expr]:
    return [
        (pl.col(f"log1p_ts_velo_rol_mean_{window}") * pl.col(f"px_pct_rol_sum_{window}"))
        .alias(f"px_velo_rol_mean_{window}"),

        (pl.col(f"log1p_sum_sz_{window}") * pl.col(f"px_pct_rol_sum_{window}"))
        .alias(f"sum_sz_px_pct_rol_sum_{window}"),

        (pl.col("z_log1p_sum_open_interest_value") - pl.col(f"px_pct_rol_sum_{window}"))
        .alias(f"oi_px_diff_{window}"),
    ]

def tick_advanced_factors(window: int) -> List[Expr]:
    return [
        (pl.col("z_oi_up_divergence") + pl.col("z_oi_dn_divergence"))
        .alias("oi_di"),

        (pl.col("oi_up_divergence_long_term") + pl.col("oi_dn_divergence_long_term"))
        .alias("oi_di_long_term"),

        (pl.col("oi_up_divergence_short_term") + pl.col("oi_dn_divergence_short_term"))
        .alias("oi_di_short_term"),

        (pl.col(f"bs_imba_rol_sum_{window}") - pl.col(f"px_pct_rol_sum_{window}"))
        .alias("taker_px_pct_diff"),

        (pl.col(f"z_px_pct_rol_sum_{window}") * pl.col(f"log1p_sum_sz_{window}"))
        .alias("factor_impact_momentum"),

        (pl.col(f"z_px_pct_rol_sum_{window}") / (pl.col(f"log1p_sum_sz_{window}") + EPSILON))
        .alias("factor_impact_sensitivity"),

        (pl.col(f"z_px_pct_rol_sum_{window}") - pl.col(f"z_bs_ratio_rol_mean_{window}"))
        .alias("factor_orderflow_sz_momentum"),

        (pl.col(f"z_px_pct_rol_sum_{window}") - pl.col(f"z_bs_imba_rol_mean_{window}"))
        .alias("factor_orderflow_sz_sensitivity"),
    ]

def trend_matrix_factors(col: str, window: int) -> List[Expr]:
    return [
        (pl.col(f"px_max_{window}") - pl.col(col))
        .clip(lower_bound=0.0)
        .alias(f"{col}_drawdown"),


        (pl.col(col) - pl.col(f"px_min_{window}"))
        .clip(lower_bound=0.0)
        .alias(f"{col}_rebound"),
    ]

def micro_structure_factors(window: int) -> List[Expr]:
    return [
        (pl.col("z_px_drawdown") + pl.col("z_px_rebound") / 2.0)
        .alias("px_dd_rb"),

        pl
        .when(pl.col(f"z_px_pct_rol_sum_{window}") > 1.0)
        .then((pl.col(f"px_max_{window}") - pl.col("px")) / pl.col("px_range"))
        .when(pl.col(f"z_px_pct_rol_sum_{window}") < -1.0)
        .then((pl.col("px") - pl.col(f"px_min_{window}")) / pl.col("px_range"))
        .otherwise(0.0)
        .alias(f"micro_trend_factor_{window}")
    ]

def alt_tick_composite_factors() -> List[Expr]:
    return [
        (pl.col("z_factor_orderflow_sz_momentum") * pl.col("factor_triplet"))
        .alias("factor_order_momentum_divergence"),

        (pl.col("z_factor_orderflow_sz_sensitivity") * pl.col("z_factor_triplet"))
        .alias("factor_order_sentiment_divergence"),
    ]


def oi_px_correlation_factors(price_col: str, oi_value_col: str, window: int = 100) -> List[Expr]:
    corr_expr = (
        F.rolling_corr(price_col, oi_value_col, window_size=window, min_samples=1)
        .fill_null(0)
        .alias(f"corr_{price_col}_{oi_value_col}_{window}")
    )

    z_signal_expr = (
        pl.when(corr_expr > -0.7)
        .then(1)
        .otherwise(0)
        .alias(f"z_signal_{price_col}_{oi_value_col}_{window}")
    )

    return [corr_expr, z_signal_expr]

def cal_factors_with_sampled_data(
        input_df: pl.DataFrame,
        window: int,
) -> pl.DataFrame:
    factors_df = (
        input_df
        .with_columns(
            tick_basic_factors(window),
        )
        .with_columns(
            trade_size_basic_factors(window),
        )
        .with_columns([
            (pl.col(f"px_max_{window}") - pl.col(f"px_min_{window}"))
            .clip(lower_bound=0.0)
            .alias("px_range"),

            pl.col("ts_duration")
            .clip(lower_bound=0.0)
            .log1p()
            .alias(f"log1p_ts_velo_rol_mean_{window}"),

            pl.col("oi_sum_open_interest_value")
            .clip(lower_bound=0.0)
            .log1p()
            .alias("log1p_sum_open_interest_value"),
        ])
        .with_columns([
            rsi_expr(f"log1p_sum_sz_{window}", window),
            rsi_expr(f"log1p_ts_velo_rol_mean_{window}", window),
            rsi_expr("log1p_sum_open_interest_value", window),
        ])
        .with_columns([
            z_score_expr(f"px_pct_rol_sum_{10}", window),
            z_score_expr(f"px_pct_rol_sum_{20}", window),
            z_score_expr(f"px_pct_rol_sum_{40}", window),
            z_score_expr(f"px_pct_rol_sum_{80}", window),
            z_score_expr(f"px_pct_rol_sum_{160}", window),
            z_score_expr(f"px_pct_rol_sum_{window}", window),
            z_score_expr(f"bs_ratio_rol_mean_{window}", window),
            z_score_expr(f"bs_imba_rol_mean_{window}", window),
            z_score_expr(f"bs_imba_rol_sum_{window}", window),
            z_score_expr(f"log1p_ts_velo_rol_mean_{window}", window),
            z_score_expr("log1p_sum_open_interest_value", window),
            z_score_expr(f"log1p_sum_sz_{window}_rsi_{window}", window),
            z_score_expr(f"log1p_ts_velo_rol_mean_{window}_rsi_{window}", window),
            z_score_expr(f"log1p_sum_open_interest_value_rsi_{window}", window),

        ])
        .with_columns(
            oi_div_factors(window),
        )
        .with_columns(
            log_feat_scaling_factors(window),
        )
        .with_columns([
            z_score_expr("oi_up_divergence", window),
            z_score_expr("oi_dn_divergence", window),
            z_score_expr("oi_up_divergence_long_term", window),
            z_score_expr("oi_dn_divergence_long_term", window),
            z_score_expr("oi_up_divergence_short_term", window),
            z_score_expr("oi_dn_divergence_short_term", window),
            z_score_expr(f"sum_sz_px_pct_rol_sum_{window}", window),
            z_score_expr(f"px_velo_rol_mean_{window}", window),
            z_score_expr(f"oi_px_diff_{window}", window),
        ])
        .with_columns(
            tick_advanced_factors(window),
        )
        .with_columns([
            z_score_expr("oi_di", window),
            z_score_expr("oi_di_long_term", window),
            z_score_expr("taker_px_pct_diff", window),
            z_score_expr("factor_impact_momentum", window),
            z_score_expr("factor_impact_sensitivity", window),
            z_score_expr("factor_orderflow_sz_momentum", window),
            z_score_expr("factor_orderflow_sz_sensitivity", window),
        ])
        .with_columns(
            alt_tick_composite_factors(),
        )
        .with_columns(
            oi_px_correlation_factors("px", "oi_sum_open_interest_value", window),
        )
        .with_columns([
            z_score_expr("factor_order_momentum_divergence", window),
            z_score_expr("factor_order_sentiment_divergence", window),
            z_score_expr(f"corr_px_oi_sum_open_interest_value_{window}", window),
        ])
        .with_columns([
            (pl.col(f"z_px_pct_rol_sum_{window}") * pl.col(f"z_signal_px_oi_sum_open_interest_value_{window}"))
            .alias("z_px_oi_corr_activation")
        ])
        .with_columns(
            risk_factor_expr("px", window),
        )
        .with_columns(
            trend_matrix_expr("px", window),
        )
        .with_columns([
            z_score_expr("px_risk_factor", window),
            z_score_expr("px_drawdown", window),
            z_score_expr("px_rebound", window),
        ])
        .with_columns(
            micro_structure_factors(window),
        )
        .with_columns([
            z_score_expr("px_dd_rb", window),
            z_score_expr(f"micro_trend_factor_{window}", window),
        ])
        .drop_nulls()
    )

    return factors_df
