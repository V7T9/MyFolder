import os
import webbrowser

import pandas as pd
import vectorbt as vbt


SYMBOL = "BTCUSDT"
LIMIT = None
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FEES = 0.001       # 0.1%
SLIPPAGE = 0.0005  # 0.05%


def get_klines(interval, limit=None):
    """读取本地 data 目录中的 K 线 CSV 数据，默认使用全部数据"""
    csv_map = {
        "1h": os.path.join(DATA_DIR, "BTCUSDT_1h.csv"),
        "4h": os.path.join(DATA_DIR, "BTCUSDT_4h.csv"),
    }

    csv_path = csv_map.get(interval)
    if csv_path is None:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(csv_map.keys())}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("open_time").reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).copy()
    df = df.set_index("open_time")
    df.index.name = "time"

    return df


def main():

    # ============================================================
    # 1. 获取 Binance 原生 1H / 4H K线
    # ============================================================

    print("正在加载本地 CSV 数据...")

    df_1h = get_klines("1h")
    df_4h = get_klines("4h")

    print()
    print("1H 数据：")
    print(f"  数量: {len(df_1h)}")
    print(f"  开始: {df_1h.index[0]}")
    print(f"  结束: {df_1h.index[-1]}")

    print()
    print("4H 数据：")
    print(f"  数量: {len(df_4h)}")
    print(f"  开始: {df_4h.index[0]}")
    print(f"  结束: {df_4h.index[-1]}")

    # ============================================================
    # 2. 1H MACD
    # ============================================================

    close_1h = df_1h["close"]
    open_1h = df_1h["open"]

    macd_1h = vbt.MACD.run(
        close_1h,
        fast_window=12,
        slow_window=26,
        signal_window=9,
    )

    # 1H 金叉
    macd_1h_buy = (
        macd_1h.macd_crossed_above(macd_1h.signal)
        .fillna(False)
        .astype(bool)
    )

    # 1H 死叉
    macd_1h_sell = (
        macd_1h.macd_crossed_below(macd_1h.signal)
        .fillna(False)
        .astype(bool)
    )

    # ============================================================
    # 3. 4H MACD
    # ============================================================

    close_4h = df_4h["close"]

    macd_4h = vbt.MACD.run(
        close_4h,
        fast_window=12,
        slow_window=26,
        signal_window=9,
    )

    # 4H 当前是否处于 MACD 多头状态
    trend_4h = (
        macd_4h.macd > macd_4h.signal
    )

    # 4H 死叉
    macd_4h_sell = (
        macd_4h.macd_crossed_below(macd_4h.signal)
        .fillna(False)
        .astype(bool)
    )

    # ============================================================
    # 4. 把 4H MACD 状态映射到 1H
    #
    # 注意：
    # Binance 的 4H 时间戳是 K线开始时间。
    #
    # 因此一根 4H K线在自己的 open_time 时刻还没有收盘，
    # 不能让 1H 策略提前知道它最终的 MACD。
    #
    # 我们把 4H 状态向后移动一个 1H K线，
    # 确保只使用已经完成的 4H 信息。
    # ============================================================

    trend_4h_on_1h = (
        trend_4h
        .reindex(df_1h.index, method="ffill")
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # 4H 死叉同样映射到 1H
    macd_4h_sell_on_1h = (
        macd_4h_sell
        .reindex(df_1h.index, method="ffill")
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # ============================================================
    # 5. 生成策略信号
    #
    # 买入：
    #   4H MACD处于多头
    #   AND
    #   1H MACD刚刚金叉
    #
    # 卖出：
    #   1H MACD死叉
    #   OR
    #   4H MACD死叉
    # ============================================================

    raw_entries = (
        macd_1h_buy
        & trend_4h_on_1h
    )

    raw_exits = (
        macd_1h_sell
        | macd_4h_sell_on_1h
    )

    # ============================================================
    # 6. 下一根 1H K线成交
    #
    # 信号在这一根K线收盘后产生，
    # 下一根K线开盘执行。
    # ============================================================

    entries = (
        raw_entries
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    exits = (
        raw_exits
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # ============================================================
    # 7. 回测
    #
    # 使用下一根K线的开盘价作为成交基础价格。
    # VectorBT 自动加入手续费和滑点。
    # ============================================================

    pf = vbt.Portfolio.from_signals(
        open_1h,
        entries,
        exits,
        init_cash=100,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1h",
    )

    # ============================================================
    # 8. 输出结果
    # ============================================================

    print()
    print("=" * 50)
    print(" BTC Quant - 4H MACD + 1H MACD")
    print("=" * 50)

    print(f"初始资金: {float(pf.init_cash):.2f} USDT")
    print(f"最终资金: {float(pf.final_value()):.2f} USDT")
    print(f"总收益率: {float(pf.total_return()) * 100:.2f}%")
    print(f"交易次数: {int(pf.trades.count())}")
    print(f"最大回撤: {float(pf.max_drawdown()) * 100:.2f}%")

    print()
    print(f"手续费: {FEES * 100:.3f}%")
    print(f"滑点:   {SLIPPAGE * 100:.3f}%")

    # ============================================================
    # 9. 输出实际交易记录
    # ============================================================

    print()
    print("=" * 50)
    print("交易记录")
    print("=" * 50)

    trades = pf.trades.records_readable

    if len(trades) > 0:
        print(trades)
    else:
        print("没有发生交易。")

    # ============================================================
    # 10. 画图
    # ============================================================

    fig = pf.plot()

    output_path = os.path.abspath(
        "btc_quant_4h_1h_macd.html"
    )

    fig.write_html(
        output_path,
        auto_open=False,
    )

    print()
    print(f"图表已保存到:")
    print(output_path)

    webbrowser.open_new_tab(
        f"file://{output_path}"
    )

    input("\n按回车键退出...")


if __name__ == "__main__":
    main()