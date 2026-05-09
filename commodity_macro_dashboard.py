"""
商品与宏观指标汇总绘图脚本。

依赖:
- akshare
- pandas
- matplotlib
- requests
- beautifulsoup4

输出:
- 所有 CSV 和图片均保存到脚本同级目录下的 `output/commodity_macro` 文件夹

当前数据口径:
- 能源类: WTI 原油
- 贵金属: 黄金、白银
- 黑色系: 螺纹钢主力连续、铁矿石主力连续
- 农产品: 大豆、玉米
- 铜金比: 铜收盘价 / 金收盘价
- 10 年期美债收益率: 美国国债收益率10年
- CRB 指数: 公开网页可获取的近期历史序列
"""

import math
import re
from pathlib import Path

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


LOOKBACK_YEARS = 3
HISTORY_START = pd.Timestamp("2000-01-01")
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "commodity_macro"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def trim_lookback(df: pd.DataFrame, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    filtered = df[pd.to_datetime(df["date"]) >= cutoff]
    return filtered.sort_values("date").dropna().reset_index(drop=True)


def trim_since(df: pd.DataFrame, start: pd.Timestamp = HISTORY_START) -> pd.DataFrame:
    filtered = df[pd.to_datetime(df["date"]) >= start.normalize()]
    return filtered.sort_values("date").dropna().reset_index(drop=True)


def normalize_series(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    first_value = result["value"].iloc[0]
    result["value"] = result["value"] / first_value * 100
    return result


def to_standard_frame(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    result = df[[date_col, value_col]].copy()
    result.columns = ["date", "value"]
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.dropna().sort_values("date").reset_index(drop=True)


def fetch_foreign_futures(symbol: str) -> pd.DataFrame:
    yahoo_map = {
        "CL": "CL=F",
        "GC": "GC=F",
        "SI": "SI=F",
        "HG": "HG=F",
        "S": "ZS=F",
        "C": "ZC=F",
    }
    ticker = yahoo_map.get(symbol)
    if ticker is not None:
        try:
            history = yf.Ticker(ticker).history(start=HISTORY_START.strftime("%Y-%m-%d"), auto_adjust=False)
            if not history.empty:
                data = history[["Close"]].copy().reset_index()
                date_column = "Date" if "Date" in data.columns else data.columns[0]
                data = data.rename(columns={date_column: "date", "Close": "value"})
                data["date"] = pd.to_datetime(data["date"]).dt.date
                data["value"] = pd.to_numeric(data["value"], errors="coerce")
                return trim_since(data[["date", "value"]])
        except Exception:
            pass

    raw_df = ak.futures_foreign_hist(symbol=symbol)
    return trim_since(to_standard_frame(raw_df, "date", "close"))


def fetch_domestic_futures(symbol: str) -> pd.DataFrame:
    raw_df = ak.futures_zh_daily_sina(symbol=symbol)
    return trim_lookback(to_standard_frame(raw_df, "date", "close"))


def fetch_us10y_yield() -> pd.DataFrame:
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)).strftime("%Y%m%d")
    raw_df = ak.bond_zh_us_rate(start_date=start_date)
    df = to_standard_frame(raw_df, "日期", "美国国债收益率10年")
    return df.dropna().reset_index(drop=True)


def fetch_crb_recent_series() -> pd.DataFrame:
    url = "https://ru.stockq.org/index/CRB.php"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 12:
        raise ValueError("未找到 CRB 历史表格")

    table_text = " ".join(tables[11].get_text(" ", strip=True).split())
    pattern = re.compile(r"(\d{4}/\d{2}/\d{2})\s+([\d.]+)\s+([+-]?[\d.]+)\s+%")
    rows = pattern.findall(table_text)
    if not rows:
        raise ValueError("未解析到 CRB 历史数据")

    df = pd.DataFrame(rows, columns=["date", "value", "change_pct"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)


def build_copper_gold_ratio(copper_df: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        copper_df.rename(columns={"value": "copper"}),
        gold_df.rename(columns={"value": "gold"}),
        on="date",
        how="inner",
    )
    merged["value"] = merged["copper"] / merged["gold"]
    return merged[["date", "value"]].dropna().reset_index(drop=True)


def save_series_csv(df: pd.DataFrame, slug: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{slug}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def plot_grouped_categories(groups: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for index, group in enumerate(groups):
        ax = axes[index]
        for series in group["series"]:
            df = normalize_series(series["df"])
            ax.plot(df["date"], df["value"], linewidth=1.8, label=series["label"])

        ax.set_title(group["title"], fontsize=13)
        ax.set_xlabel("日期")
        ax.set_ylabel("归一化指数 (起点=100)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    fig.suptitle("大宗商品分类走势汇总", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    image_path = OUTPUT_DIR / "commodity_category_dashboard.png"
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    return image_path


def plot_macro_indicators(indicators: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(indicators)
    cols = 2
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4.6))
    axes = axes.flatten()

    for index, item in enumerate(indicators):
        ax = axes[index]
        ax.plot(item["df"]["date"], item["df"]["value"], linewidth=1.8, color=item.get("color", "#1f77b4"))
        ax.set_title(item["title"], fontsize=13)
        ax.set_xlabel("日期")
        ax.set_ylabel(item["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45)

    for index in range(total, len(axes)):
        axes[index].axis("off")

    fig.suptitle("商品相关宏观指标走势", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    image_path = OUTPUT_DIR / "macro_indicator_dashboard.png"
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    return image_path


def main() -> None:
    configure_matplotlib()

    print("正在获取海外商品数据...")
    oil_df = fetch_foreign_futures("CL")
    gold_df = fetch_foreign_futures("GC")
    silver_df = fetch_foreign_futures("SI")
    copper_df = fetch_foreign_futures("HG")
    soybean_df = fetch_foreign_futures("S")
    corn_df = fetch_foreign_futures("C")

    print("正在获取国内黑色系数据...")
    rebar_df = fetch_domestic_futures("RB0")
    iron_ore_df = fetch_domestic_futures("I0")

    print("正在获取宏观指标...")
    us10y_df = fetch_us10y_yield()
    crb_df = fetch_crb_recent_series()
    copper_gold_ratio_df = build_copper_gold_ratio(copper_df, gold_df)

    series_to_save = {
        "oil_wti_daily": oil_df,
        "gold_daily": gold_df,
        "silver_daily": silver_df,
        "copper_daily": copper_df,
        "soybean_daily": soybean_df,
        "corn_daily": corn_df,
        "rebar_daily": rebar_df,
        "iron_ore_daily": iron_ore_df,
        "crb_index_recent": crb_df,
        "copper_gold_ratio_daily": copper_gold_ratio_df,
        "us10y_yield_daily": us10y_df,
    }

    for slug, df in series_to_save.items():
        save_series_csv(df, slug)

    grouped_categories = [
        {
            "title": "能源类（石油）",
            "series": [
                {"label": "WTI原油", "df": oil_df},
            ],
        },
        {
            "title": "贵金属类（黄金、白银）",
            "series": [
                {"label": "黄金", "df": gold_df},
                {"label": "白银", "df": silver_df},
            ],
        },
        {
            "title": "黑色系（螺纹钢、铁矿石）",
            "series": [
                {"label": "螺纹钢主力连续", "df": rebar_df},
                {"label": "铁矿石主力连续", "df": iron_ore_df},
            ],
        },
        {
            "title": "农产品（大豆、玉米）",
            "series": [
                {"label": "大豆", "df": soybean_df},
                {"label": "玉米", "df": corn_df},
            ],
        },
    ]

    macro_indicators = [
        {
            "title": "CRB 指数走势",
            "ylabel": "指数点位",
            "df": crb_df,
            "color": "#8c564b",
        },
        {
            "title": "铜金比价走势",
            "ylabel": "铜价/金价",
            "df": copper_gold_ratio_df,
            "color": "#ff7f0e",
        },
        {
            "title": "10年期美债收益率走势",
            "ylabel": "%",
            "df": us10y_df,
            "color": "#2ca02c",
        },
    ]

    category_path = plot_grouped_categories(grouped_categories)
    macro_path = plot_macro_indicators(macro_indicators)

    print(f"分类汇总图已保存: {category_path}")
    print(f"宏观指标图已保存: {macro_path}")


if __name__ == "__main__":
    main()
