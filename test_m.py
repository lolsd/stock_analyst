"""
美国国债收益率自动获取 & 趋势图脚本
=====================================
数据来源：FRED（美联储经济数据库） - 完全免费
API Key：https://fred.stlouisfed.org/docs/api/api_key.html 注册即得

依赖安装：
    pip install requests pandas matplotlib

使用方式：
    python us_treasury_yield_chart.py
    python us_treasury_yield_chart.py --years 5        # 自定义年限
    python us_treasury_yield_chart.py --save chart.png # 保存为图片
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import argparse
import sys
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 配置区
# ──────────────────────────────────────────────────────────────────────────────

# ① 在此填入你的 FRED API Key（免费注册：https://fred.stlouisfed.org/docs/api/api_key.html）
FRED_API_KEY = "3d180c3fcfa0c20b61912c1d95b4ebe9"

# ② 要拉取的利率系列（FRED Series ID）
SERIES = {
    "3个月期 (DGS3MO)":  "DGS3MO",
    "2年期 (DGS2)":      "DGS2",
    "5年期 (DGS5)":      "DGS5",
    "10年期 (DGS10)":    "DGS10",
    "30年期 (DGS30)":    "DGS30",
}

# ③ 颜色映射（与系列顺序对应）
COLORS = {
    "3个月期 (DGS3MO)": "#854F0B",
    "2年期 (DGS2)":     "#185FA5",
    "5年期 (DGS5)":     "#993556",
    "10年期 (DGS10)":   "#0F6E56",
    "30年期 (DGS30)":   "#633806",
}

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ──────────────────────────────────────────────────────────────────────────────
# 数据获取
# ──────────────────────────────────────────────────────────────────────────────

def fetch_series(series_id: str, start: str, end: str) -> pd.Series:
    """从 FRED 获取单个数据系列，返回 pd.Series（index=日期，值=收益率）"""
    params = {
        "series_id":   series_id,
        "api_key":     FRED_API_KEY,
        "file_type":   "json",
        "observation_start": start,
        "observation_end":   end,
        "frequency":   "d",          # 日频
        "aggregation_method": "avg",
    }
    resp = requests.get(FRED_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "observations" not in data:
        raise ValueError(f"FRED 返回异常：{data.get('error_message', '未知错误')}")

    records = {
        obs["date"]: float(obs["value"])
        for obs in data["observations"]
        if obs["value"] != "."       # FRED 用 "." 表示缺失值
    }
    s = pd.Series(records, name=series_id)
    s.index = pd.to_datetime(s.index)
    return s


def fetch_all(years: int = 2) -> pd.DataFrame:
    """拉取所有系列，合并为 DataFrame"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    print(f"\n正在从 FRED 拉取数据（{start} → {end}）…")
    frames = {}
    for label, sid in SERIES.items():
        try:
            print(f"  [{sid}] {label} … ", end="", flush=True)
            frames[label] = fetch_series(sid, start, end)
            print("✓")
        except Exception as e:
            print(f"✗ 跳过（{e}）")

    if not frames:
        raise RuntimeError("所有系列均获取失败，请检查 API Key 和网络连接。")

    df = pd.DataFrame(frames)
    df = df.sort_index().interpolate(method="time").dropna(how="all")
    print(f"\n数据已就绪：{len(df)} 个交易日，{len(df.columns)} 个系列\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────────────────────────────────────

def compute_spread(df: pd.DataFrame) -> Optional[pd.Series]:
    """计算 10年期 - 2年期 利差"""
    col10 = "10年期 (DGS10)"
    col2  = "2年期 (DGS2)"
    if col10 in df.columns and col2 in df.columns:
        return (df[col10] - df[col2]).rename("10Y-2Y利差")
    return None


def add_recession_bands(ax, df: pd.DataFrame):
    """在图上标注重要历史事件（可自行扩充）"""
    events = [
        ("2022-03-16", "首次加息"),
        ("2023-07-26", "最后加息"),
    ]
    for date_str, label in events:
        dt = pd.Timestamp(date_str)
        if df.index.min() <= dt <= df.index.max():
            ax.axvline(dt, color="#888780", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(dt, ax.get_ylim()[1] * 0.98, label,
                    fontsize=7.5, color="#5F5E5A", ha="center", va="top",
                    rotation=90, rotation_mode="anchor")


def plot_yields(df: pd.DataFrame, spread: Optional[pd.Series], save_path: Optional[str] = None):
    """绘制主图（收益率折线）+ 副图（2Y10Y利差）"""
    has_spread = spread is not None
    n_rows = 2 if has_spread else 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 9 if has_spread else 6),
        gridspec_kw={"height_ratios": [3, 1.2] if has_spread else [1]},
        sharex=True
    )
    if n_rows == 1:
        axes = [axes]

    fig.patch.set_facecolor("#FAFAF8")

    # ── 主图：收益率 ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FAFAF8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.tick_params(colors="#5F5E5A", labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("收益率 (%)", fontsize=10, color="#5F5E5A", labelpad=8)
    ax.grid(axis="y", color="#E8E6E0", linewidth=0.6, linestyle="-")
    ax.grid(axis="x", color="#E8E6E0", linewidth=0.4, linestyle="--")

    for label, col in zip(SERIES.keys(), df.columns):
        color = COLORS.get(label, "#888780")
        lw = 2.0 if "10年期" in label or "2年期" in label else 1.4
        ls = "--" if "30年期" in label else "-"
        ax.plot(df.index, df[col], color=color, linewidth=lw,
                linestyle=ls, label=label, alpha=0.92)

    # 标注最新值
    for label, col in zip(SERIES.keys(), df.columns):
        last_val = df[col].dropna().iloc[-1]
        color = COLORS.get(label, "#888780")
        ax.annotate(
            f"{last_val:.2f}%",
            xy=(df.index[-1], last_val),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold", va="center"
        )

    add_recession_bands(ax, df)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85,
              frameon=True, edgecolor="#D3D1C7", facecolor="#FAFAF8")

    title_date = df.index[-1].strftime("%Y年%m月%d日")
    ax.set_title(f"美国国债收益率走势    （截至 {title_date}）",
                 fontsize=13, fontweight="bold", color="#1A1A2E", pad=12, loc="left")

    # ── 副图：10Y-2Y 利差 ─────────────────────────────────────────────
    if has_spread and len(axes) > 1:
        ax2 = axes[1]
        ax2.set_facecolor("#FAFAF8")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.spines[["left", "bottom"]].set_color("#D3D1C7")
        ax2.tick_params(colors="#5F5E5A", labelsize=9)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.2f}%"))
        ax2.set_ylabel("利差 (%)", fontsize=9, color="#5F5E5A", labelpad=8)
        ax2.grid(axis="y", color="#E8E6E0", linewidth=0.5)
        ax2.axhline(0, color="#888780", linewidth=0.9, linestyle="-")

        pos = spread.clip(lower=0)
        neg = spread.clip(upper=0)
        ax2.fill_between(spread.index, pos, 0, alpha=0.55, color="#0F6E56", label="正利差（正常）")
        ax2.fill_between(spread.index, neg, 0, alpha=0.55, color="#A32D2D", label="负利差（倒挂）")
        ax2.plot(spread.index, spread, color="#185FA5", linewidth=1.2, alpha=0.8)

        last_sp = spread.dropna().iloc[-1]
        sp_color = "#A32D2D" if last_sp < 0 else "#0F6E56"
        sp_label = f"当前：{last_sp:+.2f}%  {'⚠ 曲线倒挂' if last_sp < 0 else '✓ 曲线正常'}"
        ax2.set_title("10年期 − 2年期 利差（衰退预警指标）",
                      fontsize=10, fontweight="bold", color="#1A1A2E", pad=6, loc="left")
        ax2.text(0.99, 0.92, sp_label, transform=ax2.transAxes,
                 fontsize=9, color=sp_color, ha="right", va="top", fontweight="bold")

        legend_patches = [
            Patch(color="#0F6E56", alpha=0.55, label="正利差"),
            Patch(color="#A32D2D", alpha=0.55, label="负利差（倒挂）"),
        ]
        ax2.legend(handles=legend_patches, loc="lower left", fontsize=8.5,
                   framealpha=0.85, frameon=True, edgecolor="#D3D1C7", facecolor="#FAFAF8")

    # ── X轴 ────────────────────────────────────────────────────────────
    total_days = (df.index[-1] - df.index[0]).days
    if total_days > 365 * 3:
        fmt = mdates.DateFormatter("%Y")
        loc = mdates.YearLocator()
    elif total_days > 180:
        fmt = mdates.DateFormatter("%Y-%m")
        loc = mdates.MonthLocator(interval=3)
    else:
        fmt = mdates.DateFormatter("%m-%d")
        loc = mdates.MonthLocator(interval=1)

    axes[-1].xaxis.set_major_formatter(fmt)
    axes[-1].xaxis.set_major_locator(loc)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    # ── 数据来源标注 ────────────────────────────────────────────────────
    fig.text(0.01, 0.01, "数据来源：FRED（美联储经济数据库） | https://fred.stlouisfed.org",
             fontsize=8, color="#888780", style="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#FAFAF8")
        print(f"\n图表已保存：{save_path}")
    else:
        plt.show()

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 统计摘要
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, spread: Optional[pd.Series]):
    """打印关键数值摘要"""
    print("=" * 58)
    print("  美国国债收益率 · 数据摘要")
    print("=" * 58)
    print(f"  {'系列':<22} {'最新':>7} {'30日前':>7} {'变动':>7} {'52W高':>7} {'52W低':>7}")
    print("-" * 58)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        latest = s.iloc[-1]
        prev30 = s.iloc[-22] if len(s) > 22 else s.iloc[0]
        chg    = latest - prev30
        hi52   = s.tail(252).max()
        lo52   = s.tail(252).min()
        chg_str = f"{chg:+.2f}%"
        print(f"  {col:<22} {latest:>6.2f}% {prev30:>6.2f}% {chg_str:>7} {hi52:>6.2f}% {lo52:>6.2f}%")
    if spread is not None:
        sp = spread.dropna().iloc[-1]
        status = "⚠ 倒挂" if sp < 0 else "✓ 正常"
        print("-" * 58)
        print(f"  {'10Y-2Y 利差':<22} {sp:>+7.2f}%   {status}")
    print("=" * 58)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="美国国债收益率趋势图")
    parser.add_argument("--years", type=int, default=2,   help="历史年限（默认2年）")
    parser.add_argument("--save",  type=str, default=None, help="保存路径（如 chart.png），不指定则弹窗显示")
    args = parser.parse_args()

    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        print("\n⚠  请先填入 FRED API Key！")
        print("   注册地址：https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   注册免费，通常1分钟内收到邮件。\n")
        sys.exit(1)

    try:
        df = fetch_all(years=args.years)
    except Exception as e:
        print(f"\n✗ 数据获取失败：{e}")
        sys.exit(1)

    spread = compute_spread(df)
    print_summary(df, spread)
    plot_yields(df, spread, save_path=args.save)


if __name__ == "__main__":
    main()
