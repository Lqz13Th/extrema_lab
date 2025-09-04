import json
from pathlib import Path
from tqdm import tqdm

from feature_eng.operator.pl_operators import *

def str_to_timestamp_ms(df_str_to_ts: pl.DataFrame, time_col: str) -> pl.DataFrame:
    return df_str_to_ts.with_columns(
        (pl.col(time_col)
         .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
         .cast(pl.Int64)
        ).alias(time_col)
    )

def process_timestamp(df_ts: pl.DataFrame, time_col: str) -> pl.DataFrame | None:
    df_ts = df_ts.with_columns(
        (pl.col(time_col) / 1000).cast(pl.Int64) * 1000
    )

    return df_ts

def read_all_parquets(folder: Path, prefix: str, time_col: str) -> pl.DataFrame:
    dfs = []
    if not folder.exists():
        return pl.DataFrame()

    for file in sorted(folder.glob("*.parquet")):
        df_raw_pq = pl.read_parquet(file)
        if time_col not in df_raw_pq.columns:
            continue

        # metric str k lines 特殊处理
        if time_col == "create_time":
            df_raw_pq = str_to_timestamp_ms(df_raw_pq, time_col)
            df_raw_pq = df_raw_pq.rename({time_col: "timestamp"})

        else:
            df_raw_pq = process_timestamp(df_raw_pq, time_col)
            df_raw_pq = df_raw_pq.rename({time_col: "timestamp"})

        if prefix == "oi_":
            metrics_col_to_drop = [
                "symbol",
                "count_long_short_ratio",
                "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ]
            df_raw_pq = df_raw_pq.drop(metrics_col_to_drop)

        elif prefix == "funding_":
            funding_col_to_select = [
                "timestamp",
                "funding_interval_hours",
                "last_funding_rate",
            ]
            df_raw_pq = df_raw_pq.select(funding_col_to_select)

        elif prefix == "premium_":
            premium_col_to_select = [
                "timestamp",
                "open",
            ]
            df_raw_pq = df_raw_pq.select(premium_col_to_select)

        dfs.append(df_raw_pq)

    if dfs:
        return pl.concat(dfs).sort("timestamp")
    else:
        return pl.DataFrame(schema={"timestamp": pl.Int64})


def process_symbol(base_dir, output_dir, symbol: str):
    um_symbol = symbol if symbol.endswith("T") else symbol + "T"

    metric_folder = base_dir / "metrics" / symbol
    funding_folder = base_dir / "funding_rates" / um_symbol
    premium_folder = base_dir / "premium_index_klines" / um_symbol

    metric_df = read_all_parquets(metric_folder, "oi_", time_col="create_time")
    funding_df = read_all_parquets(funding_folder, "funding_", time_col="calc_time")
    premium_df = read_all_parquets(premium_folder, "premium_", time_col="open_time")

    metric_df = to_microseconds(metric_df, "timestamp")
    funding_df = to_microseconds(funding_df, "timestamp")
    premium_df = to_microseconds(premium_df, "timestamp")

    merged = merge_dataframes_on_timestamp(
        [metric_df, funding_df, premium_df],
        prefixes=["oi_", "funding_", "premium_"]
    )
    merged = auto_fill_dataframes_with_old_data(merged)
    merged = convert_all_to_float64_except_timestamp(merged)

    merged = merged.drop_nulls()
    merged.write_parquet(output_dir / f"{um_symbol}_merged.parquet")

if __name__ == "__main__":
    BASE_DIR = Path("binance_data")
    OUTPUT_DIR = Path("merged_alt_data")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    with open("../symbols.json", "r", encoding="utf-8") as f:
        symbols_list = json.load(f)

    for sym in tqdm(symbols_list, desc="Processing symbols"):
        process_symbol(BASE_DIR, OUTPUT_DIR, sym)

