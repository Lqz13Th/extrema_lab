import json
from tqdm import tqdm

from feature_eng.factors.pl_factors import *
from feature_eng.operator.pl_operators import *

def add_zscore(
        df_zscore: pl.DataFrame,
        exclude_cols=None,
):
    if exclude_cols is None:
        exclude_cols = [
            'timestamp',
            'funding_funding_interval_hours',
            'funding_last_funding_rate',
            'premium_funding_spread',
            'adjusted_funding_rate',
            'funding_premium',
            'premium_open',
        ]

    num_cols = [
        c for c in df_zscore.columns if c not in exclude_cols and df_zscore[c].dtype in [pl.Float64]
    ]

    zscore_exprs = [z_score_expr(c, 288)for c in num_cols]

    return df_zscore.with_columns(zscore_exprs)

def alt_factors_cal(alt_merged_df: pl.DataFrame):
    alt_merged_df = alt_merged_df.with_columns([
        (pl.col("funding_last_funding_rate") / pl.col("funding_funding_interval_hours"))
        .alias("adjusted_funding_rate"),
    ]).with_columns(
        alt_basic_transform()
    ).with_columns(
        alt_basic_factors([12, 72, 144, 288])
    ).with_columns(
        alt_basic_non_sensitive_factors(288)
    ).with_columns(
        alt_advanced_factors(144)
    )

    return add_zscore(alt_merged_df).drop_nulls()

if __name__ == "__main__":
    MERGED_DIR = "../data_proc/merged_alt_data"

    with open("../symbols.json", "r", encoding="utf-8") as f:
        symbols_list = json.load(f)

    for sym in tqdm(symbols_list, desc="Cal alt merged data"):
        um_symbol = sym + "T"
        file_path = f"{MERGED_DIR}/{um_symbol}_merged.parquet"

        df = pl.read_parquet(file_path)
        df = alt_factors_cal(df)

        if um_symbol == "ETHUSDT":
            print(df)
            df = pl.read_parquet(file_path)

            # 生成每列唯一值数量
            unique_counts = {col: df[col].n_unique() for col in df.columns}

            # 打印结果，每列是否只有一个唯一值
            for col, count in unique_counts.items():
                print(f"{col}: n_unique={count}, only_one_unique={count == 1}")

