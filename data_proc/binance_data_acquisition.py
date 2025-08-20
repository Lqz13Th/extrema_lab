import os
import json
import zipfile
import hashlib
import asyncio
import aiohttp
import aiofiles
import pandas as pd

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from enum import Enum
from tqdm import tqdm

class Frequency(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"

class Market(Enum):
    SPOT = "spot"
    FUTURES = "futures"
    OPTION = "option"

class FuturesMarket(Enum):
    UM = "um"  # USDT-Margined
    CM = "cm"  # Coin-Margined

class DataType(Enum):
    METRICS = "metrics"
    AGG_TRADES = "aggTrades"
    FUNDING_RATE = "fundingRate"
    PREMIUM_INDEX_KLINES = "premiumIndexKlines"
    BVOL_INDEX = "BVOLIndex"

async def fetch(session, url, local_path, retries=5, delay=5):
    for attempt in range(retries):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                content = await resp.read()
                async with aiofiles.open(local_path, "wb") as f_fetch:
                    await f_fetch.write(content)
                return True
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                print(f"Failed {url}: {e}")
                return False

    return None

async def verify_checksum(
        local_file: str,
        checksum_url: str,
        session: aiohttp.ClientSession
) -> bool:
    try:
        async with session.get(checksum_url) as resp:
            if resp.status != 200:
                print(f"Checksum not found: {checksum_url}")
                return False
            checksum_text = await resp.text()
            expected_sha256 = checksum_text.split()[0].strip()

        sha256_hash = hashlib.sha256()
        async with aiofiles.open(local_file, "rb") as f_checksum:
            while True:
                chunk = await f_checksum.read(8192)
                if not chunk:
                    break
                sha256_hash.update(chunk)

        file_sha256 = sha256_hash.hexdigest()
        if file_sha256 != expected_sha256:
            print(f"Checksum mismatch: {local_file}")
        return file_sha256 == expected_sha256

    except Exception as e:
        print(f"Failed to verify checksum: {e}")
        return False

async def process_task(
        session,
        symbol,
        date,
        base_url,
        data_type,
        output_dir,
        kline_period=None,
        market_type="futures",
):
    symbol_dir: str | bytes = os.path.join(output_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    if market_type == Market.OPTION:
        zip_filename = f"{symbol}-BVOLIndex-{date}.zip"
        url = f"{base_url}/{symbol}/{zip_filename}"
        checksum_url = f"{url}.CHECKSUM"

    elif kline_period:
        zip_filename = f"{symbol}-{kline_period}-{date}.zip"
        url = f"{base_url}/{symbol}/{kline_period}/{zip_filename}"
        checksum_url = f"{url}.CHECKSUM"

    elif data_type == DataType.METRICS:
        zip_filename = f"{symbol}_PERP-{data_type.value}-{date}.zip"
        url = f"{base_url}/{symbol}_PERP/{zip_filename}"
        checksum_url = f"{url}.CHECKSUM"

    else:
        zip_filename = f"{symbol}-{data_type.value}-{date}.zip"
        url = f"{base_url}/{symbol}/{zip_filename}"
        checksum_url = f"{url}.CHECKSUM"

    local_zip_path = os.path.join(symbol_dir, zip_filename)
    parquet_path = os.path.join(symbol_dir, zip_filename.replace(".zip", ".parquet"))

    if os.path.exists(parquet_path):
        return True

    if not await fetch(session, url, local_zip_path):
        return False

    if not await verify_checksum(local_zip_path, checksum_url, session):
        os.remove(local_zip_path)
        print(f"Checksum mismatch {zip_filename}")
        return False

    try:
        with zipfile.ZipFile(local_zip_path, "r") as z:
            for file in z.namelist():
                with z.open(file) as csvfile:
                    df = pd.read_csv(csvfile)
                    df.to_parquet(parquet_path, index=False)
        os.remove(local_zip_path)
        return True
    except Exception as e:
        print(f"Error processing {zip_filename}: {e}")
        if os.path.exists(local_zip_path):
            os.remove(local_zip_path)
        return False

async def download_binance_data_async(
    symbols: list[str],
    market: Market | FuturesMarket,
    data_type,
    freq,
    start_date: str,
    end_date: str,
    output_dir: str,
    kline_period: str | None,
    concurrency: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)

    if freq == Frequency.MONTHLY:
        start_date = start_date[:7]
        end_date = end_date[:7]
        start = datetime.strptime(start_date, "%Y-%m")
        end = datetime.strptime(end_date, "%Y-%m")
        dates = []
        current = start

        while current <= end:
            dates.append(current.strftime("%Y-%m"))
            current += relativedelta(months=1)

    else:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]

    if market == Market.OPTION:
        base_url = f"https://data.binance.vision/data/option/{freq.value}/{data_type.value}"

    else:
        base_url = f"https://data.binance.vision/data/futures/{market.value}/{freq.value}/{data_type.value}"

    tasks = [(symbol, date, base_url, data_type, output_dir) for symbol in symbols for date in dates]

    task_name = "Downloading " + getattr(data_type, "value", str(data_type))
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        for symbol, date, base_url, data_type, output_dir in tqdm(tasks, desc=task_name, unit="file"):
            result = await process_task(
                session,
                symbol,
                date,
                base_url,
                data_type,
                output_dir,
                kline_period,
                str(market.value)
            )
            results.append(result)
        return results

if __name__ == "__main__":
    with open("../symbols.json", "r", encoding="utf-8") as f:
        symbols_list = json.load(f)

    symbols_list = ["BTCUSD", "ETHUSD"]
    start_date_str = "2022-01-01"
    end_date_str = "2025-07-31"
    concurrency_numbs = 50

    # download USD metrics
    asyncio.run(download_binance_data_async(
        symbols=symbols_list,
        market=FuturesMarket.CM,
        data_type=DataType.METRICS,
        freq=Frequency.DAILY,
        start_date=start_date_str,
        end_date=end_date_str,
        output_dir="./binance_data/metrics",
        kline_period=None,
        concurrency=concurrency_numbs,
    ))


    symbols_list_usdt = [s + "T" for s in symbols_list]

    # download funding rates
    asyncio.run(download_binance_data_async(
        symbols=symbols_list_usdt,
        market=FuturesMarket.UM,
        data_type=DataType.FUNDING_RATE,
        freq=Frequency.MONTHLY,
        start_date=start_date_str,
        end_date=end_date_str,
        output_dir="./binance_data/funding_rates",
        kline_period=None,
        concurrency=concurrency_numbs,
    ))

    # download premium index klines
    asyncio.run(download_binance_data_async(
        symbols=symbols_list_usdt,
        market=FuturesMarket.UM,
        data_type=DataType.PREMIUM_INDEX_KLINES,
        freq=Frequency.MONTHLY,
        start_date=start_date_str,
        end_date=end_date_str,
        output_dir="./binance_data/premium_index_klines",
        kline_period='15m',
        concurrency=concurrency_numbs,
    ))

    # download aggregate trades
    asyncio.run(download_binance_data_async(
        symbols=symbols_list_usdt,
        market=FuturesMarket.UM,
        data_type=DataType.AGG_TRADES,
        freq=Frequency.DAILY,
        start_date=start_date_str,
        end_date=end_date_str,
        output_dir="./binance_data/aggregate_trades",
        kline_period=None,
        concurrency=concurrency_numbs,
    ))

    # # download binance option volatility index
    # asyncio.run(download_binance_data_async(
    #     symbols=["BTCBVOLUSDT", "ETHBVOLUSDT"],
    #     market=Market.OPTION,
    #     data_type=DataType.BVOL_INDEX,
    #     freq=Frequency.DAILY,
    #     start_date=start_data_str,
    #     end_date=end_date_str,
    #     output_dir="./binance_data/BVOL",
    #     kline_period=None,
    #     concurrency=concurrency_numbs,
    # ))