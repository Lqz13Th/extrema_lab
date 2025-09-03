import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta

def validate_klines(df: pd.DataFrame, interval: str) -> bool:
    if df.empty:
        print("Empty DataFrame, skip validation.")
        return False

    # Map interval strings to milliseconds
    interval_map = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
    }
    step = interval_map.get(interval)
    if not step:
        print(f"Interval {interval} not supported in validation.")
        return False

    diffs = df["open_time"].diff().dropna()
    invalid = diffs[diffs != step]

    if not invalid.empty:
        print(f"Validation failed: {len(invalid)} gaps detected (expected step={step} ms).")
        print(invalid.head())
        return False

    print(f"Validation passed: {len(df)} rows with correct {interval} spacing.")
    return True

async def fetch_premium_index_klines(session: aiohttp.ClientSession, symbol: str, interval: str,
                                     start_time: int) -> pd.DataFrame:
    all_klines = []
    limit = 1500
    end_time = None
    url = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if end_time:
            params["endTime"] = end_time

        if end_time and end_time <= start_time:
            break

        try:
            async with session.get(url, params=params) as response:
                text = await response.text()
                if response.status != 200:
                    print(f"{symbol} request failed: {response.status}, {text}")
                    break

                klines = await response.json()
                if not isinstance(klines, list) or not klines:
                    print(f"{symbol} returned empty data: {text}")
                    break

                # Filter invalid records
                klines_cleaned = [k for k in klines if all(val is not None for val in k[1:5])]
                if not klines_cleaned:
                    print(f"{symbol} batch contained only invalid data, stopping.")
                    break

                filtered_klines = [k for k in klines_cleaned if k[0] >= start_time]
                all_klines.extend(filtered_klines)

                # Update end_time for next request (iterate backwards in time)
                end_time = klines_cleaned[0][0] - 1
                if klines_cleaned[0][0] < start_time:
                    break

                print(f"{symbol}: fetched {len(klines_cleaned)} rows, next end_time={end_time}")
                await asyncio.sleep(0.5)

        except aiohttp.ClientError as e:
            print(f"{symbol} network error: {e}")
            break

    # Convert to DataFrame
    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "ignore1",
        "close_time", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"
    ])

    df.drop(columns=[f"ignore{i}" for i in range(1, 7)], inplace=True)
    df.drop_duplicates(subset=["open_time"], inplace=True)
    df.sort_values(by="open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.insert(0, "symbol", symbol)

    validate_klines(df, interval)

    return df


async def get_monthly_premium_index_klines(symbols: list, interval: str) -> dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame} for the last month of data"""
    one_month_ago = datetime.now() - timedelta(days=30)
    start_time_ms = int(one_month_ago.timestamp() * 1000)

    result_dict = {}

    async with aiohttp.ClientSession() as session:
        tasks = {
            symbol: fetch_premium_index_klines(session, symbol, interval, start_time_ms)
            for symbol in symbols
        }
        results = await asyncio.gather(*tasks.values())

        for symbol, df in zip(tasks.keys(), results):
            result_dict[symbol] = df

    return result_dict


# Example usage
if __name__ == "__main__":
    symbol_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    interval = "5m"

    try:
        symbol_dfs = asyncio.run(get_monthly_premium_index_klines(symbol_list, interval))

        for sym, df in symbol_dfs.items():
            print(f"\n🔹 {sym}: {len(df)} rows")
            print(df.head())
            print(df.tail())

    except Exception as e:
        print(f"Runtime error: {e}")
