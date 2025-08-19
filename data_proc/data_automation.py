from datetime import date

from data_proc.alt_data_preprocessing import *
from data_proc.binance_data_acquisition import *

def data_auto_process(init_date: str):
    with open("../symbols.json", "r", encoding="utf-8") as f_automation:
        symbols_automation_list = json.load(f_automation)

    today = date.today()
    first_day_this_month = today.replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    last_month_end_date_str = last_day_last_month.strftime("%Y-%m-%d")
    print(last_day_last_month)

    # download USD metrics
    asyncio.run(download_binance_data_async(
        symbols=symbols_automation_list,
        market=FuturesMarket.CM,
        data_type=DataType.METRICS,
        freq=Frequency.DAILY,
        start_date=init_date,
        end_date=last_month_end_date_str,
        output_dir="./binance_data/metrics",
        kline_period=None,
        concurrency=concurrency_numbs,
    ))


    symbols_automation_list_usdt = [s + "T" for s in symbols_list]

    # download funding rates
    asyncio.run(download_binance_data_async(
        symbols=symbols_automation_list_usdt,
        market=FuturesMarket.UM,
        data_type=DataType.FUNDING_RATE,
        freq=Frequency.MONTHLY,
        start_date=init_date,
        end_date=last_month_end_date_str,
        output_dir="./binance_data/funding_rates",
        kline_period=None,
        concurrency=concurrency_numbs,
    ))

    # download premium index klines
    asyncio.run(download_binance_data_async(
        symbols=symbols_automation_list_usdt,
        market=FuturesMarket.UM,
        data_type=DataType.PREMIUM_INDEX_KLINES,
        freq=Frequency.MONTHLY,
        start_date=init_date,
        end_date=last_month_end_date_str,
        output_dir="./binance_data/premium_index_klines",
        kline_period='15m',
        concurrency=concurrency_numbs,
    ))

    # download aggregate trades
    asyncio.run(download_binance_data_async(
        symbols=symbols_automation_list_usdt,
        market=FuturesMarket.UM,
        data_type=DataType.AGG_TRADES,
        freq=Frequency.DAILY,
        start_date=init_date,
        end_date=end_date_str,
        output_dir="./binance_data/aggregate_trades",
        kline_period=None,
        concurrency=concurrency_numbs,
    ))

    base_automation_dir = Path("binance_data")
    output_automation_dir = Path("merged_alt_data")
    output_automation_dir.mkdir(exist_ok=True, parents=True)

    for sym_automation in tqdm(symbols_automation_list, desc="Processing symbols"):
        process_symbol(base_automation_dir, output_automation_dir, sym_automation)

if __name__ == "__main__":
    data_auto_process(init_date="2025-01-01")