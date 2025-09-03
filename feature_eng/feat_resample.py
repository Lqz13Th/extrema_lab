import os

from datetime import datetime, timedelta

from feature_eng.feat_alt_cal import *

def generate_dates(start_date_str: str, end_date_str: str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return dates


def data_resampling(
    start_date: str,
    end_date: str,
    threshold: float,
    output_dir: str,
    target_instruments: list,
    delay_minutes: int = 5,
    resample: bool = True
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_file_template = "../data_proc/merged_alt_data/{symbol}_merged.parquet"
    agg_trade_file_template = ("../data_proc/binance_data/aggregate_trades/"
                               "{symbol}/{symbol}-aggTrades-{date}.parquet")

    dates_list = generate_dates(start_date, end_date)

    for i, symbol in enumerate(target_instruments, 1):
        try:
            if symbol == "ETHUSDT":
                continue

            print(f"[{i}/{len(target_instruments)}] Processing instrument: {symbol}")

            alt_merged_df = pl.read_parquet(
                merged_file_template.format(symbol=symbol)
            )
            alt_merged_df = alt_factors_cal(alt_merged_df)

            alt_df = alt_merged_df.with_columns(
                (pl.col("timestamp") + delay_minutes * 60 * 1000 * 1000).alias("timestamp")
            )

            if resample:
                for date in tqdm(dates_list, desc=f"{symbol} resampling"):
                    agg_trade_df = pl.scan_parquet(agg_trade_file_template.format(symbol=symbol, date=date))

                    cols = [col_agg_trade for col_agg_trade in agg_trade_df.collect_schema().names() if col_agg_trade != '']
                    cols_to_cast = [c for c in cols if c not in ["is_buyer_maker"]]
                    agg_trade_df = (
                        agg_trade_df
                        .with_columns([
                            (pl.col("transact_time") * 1000).alias("timestamp")]
                        )
                        .with_columns([
                            pl.col(col_to_cast).cast(
                                pl.Int64 if col_to_cast == "timestamp" else
                                pl.Utf8 if col_to_cast == "symbol" else
                                pl.Float64
                            ) for col_to_cast in cols_to_cast
                        ])
                        .collect()
                    )

                    ts_min = agg_trade_df['timestamp'].min() - 1 * 1000 * 1000
                    ts_max = agg_trade_df['timestamp'].max()

                    daily_df = alt_df.filter(
                        (pl.col("timestamp") >= ts_min) &
                        (pl.col("timestamp") <= ts_max)
                    )

                    if daily_df.is_empty():
                        continue


                    feat_merged_df = merge_dataframes_on_timestamp(
                        [agg_trade_df, alt_df],
                        ["trades_", "alt_"]
                    )

                    auto_filled_df = auto_fill_dataframes_with_old_data(feat_merged_df).drop_nulls()
                    alt_cols = [c for c in auto_filled_df.columns if c.startswith("alt_")]

                    pct_sampled_data = generate_pct_bar(auto_filled_df, alt_cols, threshold)

                    symbol_folder = os.path.join(output_dir, symbol)
                    os.makedirs(symbol_folder, exist_ok=True)
                    output_file_path = os.path.join(
                        symbol_folder,
                        f"resampled_data_{symbol}_{date}_thr{threshold}.parquet"
                    )
                    pct_sampled_data.write_parquet(output_file_path)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

        merge_all_datas_for_symbol(
            symbol=symbol,
            dates_list=dates_list,
            input_dir=output_dir,
            output_dir=output_dir,
            threshold=threshold,
        )

def merge_all_datas_for_symbol(
        symbol: str,
        dates_list: list,
        input_dir: str,
        output_dir: str,
        threshold: float,
):
    symbol_folder = os.path.join(input_dir, symbol)

    target_files = [
        f"resampled_data_{symbol}_{date}_thr{threshold}.parquet"
        for date in dates_list
    ]

    df_list = []
    for file_name in target_files:
        fpath = os.path.join(symbol_folder, file_name)
        if os.path.exists(fpath):
            df_to_merge = pl.read_parquet(fpath)

            if df_to_merge.is_empty():
                print(f"[WARNING] {symbol} empty dataframe in file: {fpath} -> shape={df_to_merge.shape}")
                continue

            casted_df = df_to_merge.with_columns([
                pl.col(col_merging).cast(
                    pl.Int64 if col_merging == "timestamp" else pl.Float64
                ) for col_merging in df_to_merge.columns
            ])
            df_list.append(casted_df)
        else:
            print(f"[MISSING] file not exist: {fpath}")

    if not df_list:
        raise FileNotFoundError("unmatched file")

    merged_df = pl.concat(df_list).sort("timestamp")

    output_filename = f"{symbol}_merged_thr{threshold}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    merged_df.write_parquet(output_path)
    print(f"{symbol} merged, {merged_df.shape[0]} row, saved to: {output_path}")

def generate_pct_bar(
        input_df: pl.DataFrame,
        alt_cols: list[str],
        threshold: float,
) -> pl.DataFrame:

    last_px = input_df[0, "trades_price"]
    last_ts = input_df[0, "timestamp"]

    sampled_datas = []
    sum_buy_size = 0
    sum_sell_size = 0
    for row in input_df.iter_rows(named=True):
        ts = row['timestamp']
        px = row['trades_price']

        sz = row['trades_quantity']
        side = -1 if row['trades_is_buyer_maker'] == True else 1
        px_pct = (px - last_px) / last_px
        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold and ts - last_ts > 1_000_000:
            ts_duration = ts - last_ts

            sampled_data = {
                "timestamp": ts,
                "px": px,
                "sum_buy_sz": sum_buy_size,
                "sum_sell_sz": sum_sell_size,
                "ts_duration": ts_duration,
                "px_pct": px_pct,
                "bs_imbalance": sum_buy_size - sum_sell_size,

                **{c.replace("alt_", ""): row[c] for c in alt_cols}
            }

            sampled_datas.append(sampled_data)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0

    sampled_df = pl.DataFrame(sampled_datas)
    sampled_df = sampled_df.drop_nulls()

    return sampled_df

if __name__ == "__main__":
    OUTPUT_DIR = "../data_proc/resampled_data"

    with open("../symbols.json", "r", encoding="utf-8") as f:
        symbols_list = json.load(f)

    symbols_list_usdt = [s if s.endswith("T") else s + "T" for s in symbols_list]
    symbols_list_usdt = ["BTCUSDT"]
    data_resampling(
        start_date="2024-07-01",
        end_date="2025-08-31",
        threshold=0.0031,
        output_dir=OUTPUT_DIR,
        target_instruments=symbols_list_usdt,
        resample=True,
    )
    symbols_list_usdt = ["BNBUSDT"]
    data_resampling(
        start_date="2024-10-01",
        end_date="2025-08-31",
        threshold=0.0031,
        output_dir=OUTPUT_DIR,
        target_instruments=symbols_list_usdt,
        resample=True,
    )
    symbols_list_usdt = ["ETHUSDT"]
    data_resampling(
        start_date="2024-07-01",
        end_date="2025-08-31",
        threshold=0.0067,
        output_dir=OUTPUT_DIR,
        target_instruments=symbols_list_usdt,
        resample=True,
    )
    #
    # df = pl.read_parquet("../data_proc/resampled_data/BNBUSDT/resampled_data_BNBUSDT_2025-07-31_thr0.0067.parquet")
    # print(df)
    # df = cal_factors_with_sampled_data(df, 500)
    # print(df)
    # for col in df.columns:
    #     print(f"Column: {col}")
    #     print(df[col])  # 前5行
    #     print('-' * 40)
