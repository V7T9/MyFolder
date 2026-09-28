import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd


SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h"]

OUTPUT_DIR = "data"
BASE_URL = "https://api.binance.com/api/v3/klines"

# 最近 3 个月
DAYS = 360

# Binance 单次最多返回 1000 根
LIMIT = 1000


def download_klines(symbol, interval, start_time, end_time):
    """分页下载 Binance K线"""

    all_data = []

    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": LIMIT,
        }

        print(
            f"[{interval}] "
            f"下载: {datetime.fromtimestamp(current_start / 1000, tz=timezone.utc)}"
        )

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # 下一次从最后一根K线之后开始
        last_open_time = data[-1][0]
        current_start = last_open_time + 1

        # 如果已经没有更多数据
        if len(data) < LIMIT:
            break

        # 避免请求过快
        time.sleep(0.2)

    return all_data


def convert_to_dataframe(data):
    """将 Binance 原始 K线转换为 DataFrame"""

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)

    # 时间戳转换为 UTC 时间
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    # Binance 原始价格/成交量字段是字符串
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    df["trade_count"] = pd.to_numeric(df["trade_count"])

    return df


def main():

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    print("=" * 60)
    print("BTCUSDT Binance K线下载器")
    print("=" * 60)

    print(f"开始时间: {start}")
    print(f"结束时间: {now}")
    print()

    # 创建输出目录
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for interval in INTERVALS:

        print(f"\n========== {interval} ==========")

        raw_data = download_klines(
            SYMBOL,
            interval,
            start_ms,
            end_ms,
        )

        df = convert_to_dataframe(raw_data)

        # 根据 open_time 去重并排序
        df = (
            df
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        filename = f"{OUTPUT_DIR}/{SYMBOL}_{interval}.csv"

        df.to_csv(
            filename,
            index=False,
        )

        print()
        print(f"保存完成: {filename}")
        print(f"K线数量: {len(df)}")
        print(f"第一根: {df['open_time'].iloc[0]}")
        print(f"最后一根: {df['open_time'].iloc[-1]}")

    print("\n全部完成！🐦")


if __name__ == "__main__":
    main()