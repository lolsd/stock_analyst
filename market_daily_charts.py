"""
市场数据抓取与绘图脚本。

依赖:
- akshare
- pandas
- numpy
- matplotlib
- requests
- py-mini-racer

输出:
- 所有 CSV 和图片均保存到脚本同级目录下的 `output` 文件夹

当前数据口径:
- 纳斯达克指数、标普500: 新浪美股指数日线
- 国际金价、国际油价: 新浪外盘期货日线
- 标普500 PE: Multpl 公开 TTM PE 月度序列，扩展为日频
- 纳斯达克 PE: World PE Ratio 公开 Nasdaq 100 PE 月度序列，扩展为日频
- 沪深300 PE: 乐咕乐股指数滚动市盈率历史
- PE 分位: 历史上 PE 低于当日 PE 的天数 / 历史总天数
"""

import ast
from functools import lru_cache
import math
import re
from pathlib import Path
from typing import Optional

import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from akshare.stock_feature import stock_a_pe_and_pb as lg
from py_mini_racer import MiniRacer


LOOKBACK_YEARS = 3
PE_HISTORY_START = pd.Timestamp("2000-01-01")
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def normalize_close_frame(raw_df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("未获取到行情数据")

    df = raw_df[[date_col, close_col]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def trim_lookback(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=LOOKBACK_YEARS)
    return df[pd.to_datetime(df["date"]) >= cutoff].reset_index(drop=True)


def fetch_us_index_series(symbol: str) -> pd.DataFrame:
    # 美股指数改走新浪指数接口，避免 yfinance 频繁限流。
    raw_df = ak.index_us_stock_sina(symbol=symbol)
    df = normalize_close_frame(raw_df, "date", "close")
    return trim_lookback(df)


def fetch_foreign_futures_series(symbol: str) -> pd.DataFrame:
    # 黄金、原油等海外商品使用新浪外盘期货历史收盘价。
    raw_df = ak.futures_foreign_hist(symbol=symbol)
    df = normalize_close_frame(raw_df, "date", "close")
    return trim_lookback(df)


def fetch_cn_index_series(symbol: str) -> pd.DataFrame:
    # A 股指数使用新浪指数历史数据，当前用于沪深300。
    raw_df = ak.stock_zh_index_daily(symbol=symbol)
    df = normalize_close_frame(raw_df, "date", "close")
    return trim_lookback(df)


def normalize_value_frame(raw_df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = raw_df[[date_col, value_col]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def expand_to_daily(df: pd.DataFrame, start_date: pd.Timestamp = PE_HISTORY_START) -> pd.DataFrame:
    series = normalize_value_frame(df, "date", "value")
    if series.empty:
        raise ValueError("未获取到可扩展的 PE 数据")
    start = max(start_date.normalize(), pd.Timestamp(series["date"].min()).normalize())
    end = max(pd.Timestamp.today().normalize(), pd.Timestamp(series["date"].max()).normalize())
    full_dates = pd.date_range(start=start, end=end, freq="D")
    expanded = pd.DataFrame({"date": full_dates}).merge(series, on="date", how="left")
    expanded["value"] = expanded["value"].ffill()
    return expanded.dropna().reset_index(drop=True)


def compute_pe_percentile(df: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        raise ValueError("PE 数据为空，无法计算分位")
    sorted_values = np.sort(valid_values)
    percentile = np.searchsorted(sorted_values, values, side="left") / len(sorted_values) * 100
    result = df[["date"]].copy()
    result["value"] = percentile
    return result.dropna().reset_index(drop=True)


@lru_cache(maxsize=None)
def fetch_sp500_pe_monthly() -> pd.DataFrame:
    # 使用标普500 TTM PE 作为标普大盘估值口径。
    url = "https://www.multpl.com/s-p-500-pe-ratio"
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    match = re.search(r"let pi = (\[\[.*?\]\]);", response.text, re.S)
    if not match:
        raise ValueError("未找到标普500 PE 数据")

    payload = ast.literal_eval(match.group(1))
    date_offsets = payload[0]
    values = payload[1]
    df = pd.DataFrame(
        {
            "date": pd.Timestamp("1970-01-01") + pd.to_timedelta(date_offsets, unit="D"),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    )
    return normalize_value_frame(df, "date", "value")


@lru_cache(maxsize=None)
def fetch_nasdaq_pe_monthly() -> pd.DataFrame:
    url = "https://worldperatio.com/index/nasdaq-100/"
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    start = response.text.find("detailPE_data = [")
    end = response.text.find("detailPE_data_avg", start)
    if start == -1 or end == -1:
        raise ValueError("未找到纳斯达克 PE 数据")

    chunk = response.text[start:end]
    matches = re.findall(
        r"Date\.UTC\((\d+),\s*(\d+),\s*(\d+)\)\s*,\s*([0-9]+(?:\.[0-9]+)?)",
        chunk,
    )
    if not matches:
        raise ValueError("纳斯达克 PE 数据解析失败")

    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(int(year), int(month) + 1, int(day)),
                "value": float(value),
            }
            for year, month, day, value in matches
        ]
    )
    return normalize_value_frame(df, "date", "value")


@lru_cache(maxsize=None)
def fetch_hs300_pe_raw() -> pd.DataFrame:
    js_functions = MiniRacer()
    js_functions.eval(lg.hash_code)
    token = js_functions.call("hex", pd.Timestamp.today().date().isoformat()).lower()
    response = requests.get(
        "https://legulegu.com/api/stockdata/index-basic-pe",
        params={"token": token, "indexCode": "000300.SH"},
        timeout=20,
        **lg.get_cookie_csrf(url="https://legulegu.com/stockdata/sz50-ttm-lyr"),
    )
    response.raise_for_status()
    data = pd.DataFrame(response.json()["data"])
    parsed_date = pd.to_numeric(data["date"], errors="coerce")
    if parsed_date.notna().all():
        date_series = (
            pd.to_datetime(parsed_date.astype("int64"), unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    else:
        date_series = pd.to_datetime(data["date"], errors="coerce")
    df = pd.DataFrame({"date": date_series, "value": pd.to_numeric(data["ttmPe"], errors="coerce")})
    return normalize_value_frame(df, "date", "value")


@lru_cache(maxsize=None)
def fetch_sp500_pe_series() -> pd.DataFrame:
    return expand_to_daily(fetch_sp500_pe_monthly())


@lru_cache(maxsize=None)
def fetch_nasdaq_pe_series() -> pd.DataFrame:
    return expand_to_daily(fetch_nasdaq_pe_monthly())


@lru_cache(maxsize=None)
def fetch_hs300_pe_series() -> pd.DataFrame:
    return expand_to_daily(fetch_hs300_pe_raw())


def save_series_csv(df: pd.DataFrame, slug: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{slug}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def load_existing_series_csv(slug: str) -> pd.DataFrame:
    csv_path = OUTPUT_DIR / f"{slug}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def plot_single_series(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    slug: str,
    *,
    is_percentile: bool = False,
    latest_note: Optional[str] = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["value"], linewidth=2, color="#1f77b4")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("日期")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    if is_percentile:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.axhline(50, linestyle="--", linewidth=1, color="#94a3b8", alpha=0.8)
        ax.axhline(80, linestyle=":", linewidth=1, color="#dc2626", alpha=0.7)
        ax.axhline(20, linestyle=":", linewidth=1, color="#16a34a", alpha=0.7)
    latest_date = df["date"].iloc[-1]
    latest_value = float(df["value"].iloc[-1])
    ax.scatter([latest_date], [latest_value], color="#dc2626", s=28, zorder=3)
    if latest_note:
        ax.text(
            0.99,
            0.96,
            latest_note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffffff", "edgecolor": "#cbd5e1"},
        )
    plt.xticks(rotation=45)
    plt.tight_layout()

    image_path = OUTPUT_DIR / f"{slug}.png"
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    return image_path


def plot_dashboard(series_map: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(series_map)
    cols = 2
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4.5))
    axes = axes.flatten()

    for index, item in enumerate(series_map):
        ax = axes[index]
        ax.plot(item["df"]["date"], item["df"]["value"], linewidth=1.8)
        ax.set_title(item["title"], fontsize=13)
        ax.set_xlabel("日期")
        ax.set_ylabel(item["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45)

    for index in range(total, len(axes)):
        axes[index].axis("off")

    fig.suptitle("全球主要市场与估值指标日度走势", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    image_path = OUTPUT_DIR / "market_daily_dashboard.png"
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    return image_path


def main() -> None:
    configure_matplotlib()

    # 当前脚本的数据口径：
    # 1. 纳斯达克、标普500：新浪美股指数日线
    # 2. 国际金价、国际油价：新浪外盘期货日线
    # 3. 标普500 PE：Multpl 月度 PE 扩展为日频
    # 4. 纳斯达克 PE：World PE Ratio 月度 PE 扩展为日频
    # 5. 沪深300：新浪 A 股指数日线
    # 6. 沪深300 PE：乐咕乐股指数滚动 PE 历史
    tasks = [
        {
            "title": "纳斯达克指数每日收盘价",
            "ylabel": "指数点位",
            "slug": "nasdaq_daily",
            "fetcher": lambda: fetch_us_index_series(".IXIC"),
        },
        {
            "title": "标普500每日收盘价",
            "ylabel": "指数点位",
            "slug": "sp500_daily",
            "fetcher": lambda: fetch_us_index_series(".INX"),
        },
        {
            "title": "国际金价每日收盘价",
            "ylabel": "美元/盎司",
            "slug": "gold_daily",
            "fetcher": lambda: fetch_foreign_futures_series("GC"),
        },
        {
            "title": "国际油价每日收盘价",
            "ylabel": "美元/桶",
            "slug": "oil_daily",
            "fetcher": lambda: fetch_foreign_futures_series("CL"),
        },
        {
            "title": "标普500 PE 每日数据",
            "ylabel": "PE",
            "slug": "sp500_pe_daily",
            "fetcher": fetch_sp500_pe_series,
        },
        {
            "title": "标普500 PE 分位走势",
            "ylabel": "分位 (%)",
            "slug": "sp500_pe_percentile_daily",
            "fetcher": lambda: compute_pe_percentile(fetch_sp500_pe_series()),
            "is_percentile": True,
        },
        {
            "title": "纳斯达克 PE 每日数据",
            "ylabel": "PE",
            "slug": "nasdaq_pe_daily",
            "fetcher": fetch_nasdaq_pe_series,
        },
        {
            "title": "纳斯达克 PE 分位走势",
            "ylabel": "分位 (%)",
            "slug": "nasdaq_pe_percentile_daily",
            "fetcher": lambda: compute_pe_percentile(fetch_nasdaq_pe_series()),
            "is_percentile": True,
        },
        {
            "title": "沪深300每日收盘价",
            "ylabel": "指数点位",
            "slug": "hs300_daily",
            "fetcher": lambda: fetch_cn_index_series("sh000300"),
        },
        {
            "title": "沪深300 PE 每日数据",
            "ylabel": "PE",
            "slug": "hs300_pe_daily",
            "fetcher": fetch_hs300_pe_series,
        },
        {
            "title": "沪深300 PE 分位走势",
            "ylabel": "分位 (%)",
            "slug": "hs300_pe_percentile_daily",
            "fetcher": lambda: compute_pe_percentile(fetch_hs300_pe_series()),
            "is_percentile": True,
        },
    ]

    results = []
    for task in tasks:
        print(f"正在获取: {task['title']}")
        try:
            df = task["fetcher"]()
            print("  本次抓取成功")
        except Exception as exc:  # noqa: PERF203
            print(f"  抓取失败，回退旧数据：{exc}")
            df = load_existing_series_csv(task["slug"])
        save_series_csv(df, task["slug"])
        latest_note = None
        if task.get("is_percentile"):
            latest_note = f"今日 PE 分位: {df['value'].iloc[-1]:.2f}%"
        plot_single_series(
            df,
            task["title"],
            task["ylabel"],
            task["slug"],
            is_percentile=bool(task.get("is_percentile")),
            latest_note=latest_note,
        )
        print(df.tail())
        print("-" * 60)
        results.append(
            {
                "title": task["title"],
                "ylabel": task["ylabel"],
                "slug": task["slug"],
                "df": df,
            }
        )

    dashboard_path = plot_dashboard(results)
    print(f"汇总图已保存: {dashboard_path}")


if __name__ == "__main__":
    main()
