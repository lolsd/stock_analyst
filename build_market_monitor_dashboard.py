from __future__ import annotations

import ast
import html
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
INDICATOR_DIR = OUTPUT_DIR / "us_stock_indicators"
CATALOG_CSV = INDICATOR_DIR / "catalog.csv"
OUTPUT_HTML = OUTPUT_DIR / "market_monitor_dashboard.html"


MAJOR_MARKETS = [
    {
        "name": "纳斯达克",
        "symbol": "NASDAQ",
        "csv": OUTPUT_DIR / "nasdaq_daily.csv",
        "kind": "equity",
        "desc": "美股科技成长风向标",
    },
    {
        "name": "标普 500",
        "symbol": "S&P 500",
        "csv": OUTPUT_DIR / "sp500_daily.csv",
        "kind": "equity",
        "desc": "美股核心宽基指数",
    },
    {
        "name": "沪深 300",
        "symbol": "HS300",
        "csv": OUTPUT_DIR / "hs300_daily.csv",
        "kind": "equity",
        "desc": "中国核心资产代表",
    },
    {
        "name": "黄金",
        "symbol": "Gold",
        "csv": OUTPUT_DIR / "gold_daily.csv",
        "kind": "commodity",
        "desc": "避险与实际利率镜像",
    },
    {
        "name": "石油",
        "symbol": "WTI",
        "csv": OUTPUT_DIR / "oil_daily.csv",
        "kind": "commodity",
        "desc": "通胀与全球需求代理",
    },
    {
        "name": "美元指数",
        "symbol": "DXY",
        "csv": INDICATOR_DIR / "05_risk_stress" / "dxy" / "data.csv",
        "kind": "fx",
        "desc": "全球流动性与避险温度",
    },
]


COMMODITY_TRACKERS = [
    {"name": "CRB 指数", "symbol": "CRB", "csv": OUTPUT_DIR / "commodity_macro" / "crb_index_recent.csv", "desc": "综合商品价格指数"},
    {"name": "WTI 原油", "symbol": "WTI", "csv": OUTPUT_DIR / "commodity_macro" / "oil_wti_daily.csv", "desc": "能源价格核心锚"},
    {"name": "黄金", "symbol": "Gold", "csv": OUTPUT_DIR / "commodity_macro" / "gold_daily.csv", "desc": "避险与实际利率镜像"},
    {"name": "白银", "symbol": "Silver", "csv": OUTPUT_DIR / "commodity_macro" / "silver_daily.csv", "desc": "贵金属与工业双属性"},
    {"name": "铜", "symbol": "Copper", "csv": OUTPUT_DIR / "commodity_macro" / "copper_daily.csv", "desc": "全球工业需求晴雨表"},
    {"name": "玉米", "symbol": "Corn", "csv": OUTPUT_DIR / "commodity_macro" / "corn_daily.csv", "desc": "农产品成本与粮食链"},
    {"name": "大豆", "symbol": "Soybean", "csv": OUTPUT_DIR / "commodity_macro" / "soybean_daily.csv", "desc": "农产品价格代表"},
    {"name": "铁矿石", "symbol": "Iron Ore", "csv": OUTPUT_DIR / "commodity_macro" / "iron_ore_daily.csv", "desc": "地产与制造链代理"},
    {"name": "螺纹钢", "symbol": "Rebar", "csv": OUTPUT_DIR / "commodity_macro" / "rebar_daily.csv", "desc": "中国工业需求代理"},
    {"name": "铜金比", "symbol": "Cu/Au", "csv": OUTPUT_DIR / "commodity_macro" / "copper_gold_ratio_daily.csv", "desc": "增长与风险偏好混合信号"},
]


VALUATION_TRACKERS = [
    {
        "name": "标普 500 PE",
        "symbol": "S&P 500",
        "pe_csv": OUTPUT_DIR / "sp500_pe_daily.csv",
        "percentile_csv": OUTPUT_DIR / "sp500_pe_percentile_daily.csv",
        "desc": "标普500 TTM PE 与历史分位",
    },
    {
        "name": "纳斯达克 PE",
        "symbol": "NASDAQ",
        "pe_csv": OUTPUT_DIR / "nasdaq_pe_daily.csv",
        "percentile_csv": OUTPUT_DIR / "nasdaq_pe_percentile_daily.csv",
        "desc": "纳斯达克 PE 与历史分位",
    },
    {
        "name": "沪深 300 PE",
        "symbol": "HS300",
        "pe_csv": OUTPUT_DIR / "hs300_pe_daily.csv",
        "percentile_csv": OUTPUT_DIR / "hs300_pe_percentile_daily.csv",
        "desc": "沪深300 滚动 PE 与历史分位",
    },
]


CATEGORY_META = {
    "money_liquidity": {"title": "货币流动性", "color": "#2563eb", "bg": "#dbeafe", "icon": "M"},
    "inflation": {"title": "通胀", "color": "#d97706", "bg": "#ffedd5", "icon": "I"},
    "growth_fundamentals": {"title": "经济增长", "color": "#15803d", "bg": "#dcfce7", "icon": "G"},
    "earnings_valuation": {"title": "盈利估值", "color": "#7c3aed", "bg": "#ede9fe", "icon": "E"},
    "risk_stress": {"title": "风险压力", "color": "#dc2626", "bg": "#fee2e2", "icon": "R"},
    "flows_positioning": {"title": "资金仓位", "color": "#0f766e", "bg": "#ccfbf1", "icon": "F"},
    "technical_structure": {"title": "技术结构", "color": "#be185d", "bg": "#fce7f3", "icon": "T"},
    "sentiment_surveys": {"title": "情绪调查", "color": "#9a3412", "bg": "#ffedd5", "icon": "S"},
    "global_macro": {"title": "全球宏观", "color": "#4b5563", "bg": "#e5e7eb", "icon": "W"},
}


REPRESENTATIVE_PRIORITY = [
    "货币流动性",
    "通胀",
    "经济增长 / 基本面",
    "企业盈利与估值",
    "风险偏好与压力指标",
    "技术面与市场结构",
    "全球宏观与地缘",
]


def read_time_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    value_columns = [column for column in df.columns if column != "date"]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=value_columns, how="all").sort_values("date").reset_index(drop=True)
    return df


def latest_valid_pair(series: pd.Series) -> tuple[float | None, float | None]:
    values = series.dropna()
    if values.empty:
        return None, None
    latest = float(values.iloc[-1])
    previous = float(values.iloc[-2]) if len(values) > 1 else None
    return latest, previous


def format_value(value: float | None) -> str:
    if value is None:
        return "--"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}"
    if abs_value >= 100:
        return f"{value:,.2f}"
    if abs_value >= 10:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "--"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def relative_to_output(path_str: str | Path | None) -> str:
    if not path_str:
        return ""
    if pd.isna(path_str):
        return ""
    return Path(path_str).resolve().relative_to(OUTPUT_DIR).as_posix()


def build_chart_payload(df: pd.DataFrame) -> list[dict]:
    payload = []
    numeric_columns = [column for column in df.columns if column != "date"]
    for column in numeric_columns:
        series = df[["date", column]].dropna().copy()
        if series.empty:
            continue
        payload.append(
            {
                "name": column,
                "x": series["date"].dt.strftime("%Y-%m-%d").tolist(),
                "y": [round(float(value), 6) for value in series[column].tolist()],
            }
        )
    return payload


def build_major_market_cards() -> list[dict]:
    cards = []
    for index, item in enumerate(MAJOR_MARKETS):
        df = read_time_series(item["csv"])
        numeric_col = next(column for column in df.columns if column != "date")
        latest, previous = latest_valid_pair(df[numeric_col])
        pct = ((latest / previous) - 1) * 100 if latest is not None and previous not in (None, 0) else None
        cards.append(
            {
                "name": item["name"],
                "symbol": item["symbol"],
                "desc": item["desc"],
                "value": format_value(latest),
                "change_pct": format_delta(pct),
                "trend_class": "up" if (pct or 0) >= 0 else "down",
                "chart_id": f"major-chart-{index}",
                "chart_payload": build_chart_payload(df),
            }
        )
    return cards


def build_commodity_cards() -> list[dict]:
    cards = []
    for index, item in enumerate(COMMODITY_TRACKERS):
        df = read_time_series(item["csv"])
        numeric_col = next(column for column in df.columns if column != "date")
        latest, previous = latest_valid_pair(df[numeric_col])
        pct = ((latest / previous) - 1) * 100 if latest is not None and previous not in (None, 0) else None
        cards.append(
            {
                "name": item["name"],
                "symbol": item["symbol"],
                "desc": item["desc"],
                "value": format_value(latest),
                "change_pct": format_delta(pct),
                "trend_class": "up" if (pct or 0) >= 0 else "down",
                "chart_id": f"commodity-chart-{index}",
                "chart_payload": build_chart_payload(df),
            }
        )
    return cards


def percentile_level_class(value: float | None) -> str:
    if value is None:
        return "flat"
    if value >= 80:
        return "down"
    if value <= 20:
        return "up"
    return "flat"


def build_valuation_cards() -> list[dict]:
    cards = []
    for index, item in enumerate(VALUATION_TRACKERS):
        pe_df = read_time_series(item["pe_csv"]).rename(columns={"value": "pe"})
        percentile_df = read_time_series(item["percentile_csv"]).rename(columns={"value": "pe_percentile"})
        merged = pd.merge(pe_df, percentile_df, on="date", how="inner")
        pe_latest, pe_previous = latest_valid_pair(merged["pe"])
        percentile_latest, percentile_previous = latest_valid_pair(merged["pe_percentile"])
        pe_change = ((pe_latest / pe_previous) - 1) * 100 if pe_latest is not None and pe_previous not in (None, 0) else None
        percentile_change = (
            percentile_latest - percentile_previous
            if percentile_latest is not None and percentile_previous is not None
            else None
        )
        cards.append(
            {
                "name": item["name"],
                "symbol": item["symbol"],
                "desc": item["desc"],
                "pe_value": format_value(pe_latest),
                "pe_change": format_delta(pe_change),
                "percentile_value": f"{percentile_latest:.2f}%" if percentile_latest is not None else "--",
                "percentile_change": format_delta(percentile_change),
                "trend_class": "up" if (pe_change or 0) >= 0 else "down",
                "percentile_class": percentile_level_class(percentile_latest),
                "chart_id": f"valuation-chart-{index}",
                "chart_payload": build_chart_payload(merged),
            }
        )
    return cards


def parse_details(details_raw: str) -> dict:
    if not details_raw or details_raw == "{}":
        return {}
    try:
        return ast.literal_eval(details_raw)
    except Exception:
        return {}


def build_indicator_card(row: pd.Series) -> dict:
    details = parse_details(str(row["details"]))
    status = str(row["status"])
    data_path = relative_to_output(row.get("data_path"))

    latest_value = "--"
    latest_delta = "--"
    trend_class = "flat"
    value_label = "状态"
    chart_payload = []
    if data_path:
        df = read_time_series(Path(row["data_path"]))
        numeric_columns = [column for column in df.columns if column != "date"]
        selected_column = numeric_columns[0] if numeric_columns else None
        if selected_column:
            latest, previous = latest_valid_pair(df[selected_column])
            latest_value = format_value(latest)
            pct = ((latest / previous) - 1) * 100 if latest is not None and previous not in (None, 0) else None
            latest_delta = format_delta(pct)
            trend_class = "up" if (pct or 0) >= 0 else "down"
            value_label = selected_column
            chart_payload = build_chart_payload(df)
    else:
        latest_value = "未接通"
        latest_delta = status

    return {
        "name": str(row["indicator_name"]),
        "status": status,
        "importance": str(row["importance"]),
        "publicity": str(row["publicity"]),
        "reason": str(row["reason"]),
        "source": str(row["source"]),
        "is_representative": bool(row["representative_of_category"]),
        "note": str(row["note"] or ""),
        "error": str(row["error"] or ""),
        "data": data_path,
        "latest_value": latest_value,
        "latest_delta": latest_delta,
        "value_label": value_label,
        "trend_class": trend_class,
        "chart_payload": chart_payload,
        "chart_id": f"indicator-{row['category_slug']}-{row['indicator_slug']}",
        "calc": details.get("计算逻辑", ""),
        "meaning": details.get("代表含义", ""),
        "impact": details.get("如何影响美股", ""),
        "range": details.get("参考值", ""),
    }


def build_category_sections() -> list[dict]:
    catalog = pd.read_csv(CATALOG_CSV)
    categories = []
    for category_slug, group in catalog.groupby("category_slug", sort=False):
        meta = CATEGORY_META.get(category_slug, {"title": category_slug, "color": "#2563eb", "bg": "#eff6ff", "icon": "C"})
        rows = []
        for _, row in group.iterrows():
            rows.append(build_indicator_card(row))
        ok_statuses = {"fetched", "fetched_cached"}
        rows.sort(key=lambda item: (item["status"] not in ok_statuses, not item["is_representative"], item["name"]))
        fetched_count = sum(1 for item in rows if item["status"] in ok_statuses)
        representative = next((item for item in rows if item["is_representative"]), rows[0] if rows else None)
        categories.append(
            {
                "slug": category_slug,
                "title": meta["title"],
                "color": meta["color"],
                "bg": meta["bg"],
                "icon": meta["icon"],
                "cards": rows,
                "fetched_count": fetched_count,
                "total_count": len(rows),
                "representative": representative,
            }
        )

    order_index = {slug: index for index, slug in enumerate(CATEGORY_META)}
    categories.sort(key=lambda item: order_index.get(item["slug"], 999))
    return categories


def build_macro_summary(categories: list[dict]) -> list[dict]:
    summary_targets = {
        "money_liquidity": "货币流动性",
        "inflation": "通胀",
        "growth_fundamentals": "增长",
        "earnings_valuation": "估值",
        "risk_stress": "风险",
        "technical_structure": "技术",
        "global_macro": "全球",
    }
    items = []
    for category in categories:
        if category["slug"] not in summary_targets:
            continue
        representative = category["representative"]
        items.append(
            {
                "label": summary_targets[category["slug"]],
                "title": category["title"],
                "value": representative["latest_value"] if representative else "--",
                "delta": representative["latest_delta"] if representative else "--",
                "trend_class": representative["trend_class"] if representative else "flat",
                "name": representative["name"] if representative else "--",
                "color": category["color"],
            }
        )
    return items


def render_dashboard() -> str:
    major_cards = build_major_market_cards()
    commodity_cards = build_commodity_cards()
    valuation_cards = build_valuation_cards()
    categories = build_category_sections()
    summary_items = build_macro_summary(categories)

    nav_links = [
        ("overview", "大盘总览"),
        ("compare", "图表对比"),
        ("macro", "一级指标"),
        ("valuations", "PE估值"),
        ("commodity-trends", "大宗商品"),
        ("categories", "分类监控"),
    ]
    nav_links.extend((f"cat-{category['slug']}", category["title"]) for category in categories)

    chart_instances = []
    market_cards_html = []
    for card in major_cards:
        chart_instances.append({"id": card["chart_id"], "title": card["name"], "series": card["chart_payload"]})
        market_cards_html.append(
            f"""
            <div class="market-card chart-select-card" data-chart-id="{html.escape(card['chart_id'])}" data-compare-title="{html.escape(card['name'])}">
              <div class="market-card-top">
                <div>
                  <div class="market-symbol">{html.escape(card['symbol'])}</div>
                  <div class="market-name">{html.escape(card['name'])}</div>
                </div>
                <div class="market-desc">{html.escape(card['desc'])}</div>
              </div>
              <div class="market-value">{html.escape(card['value'])}</div>
              <div class="market-change {card['trend_class']}">{html.escape(card['change_pct'])}</div>
              <div class="line-chart chart-sm" id="{html.escape(card['chart_id'])}"></div>
            </div>
            """
        )

    macro_items_html = []
    for item in summary_items:
        macro_items_html.append(
            f"""
            <div class="signal-card">
              <div class="signal-label">{html.escape(item['label'])}</div>
              <div class="signal-name">{html.escape(item['name'])}</div>
              <div class="signal-value">{html.escape(item['value'])}</div>
              <div class="signal-delta {item['trend_class']}">{html.escape(item['delta'])}</div>
            </div>
            """
        )

    commodity_cards_html = []
    for card in commodity_cards:
        chart_instances.append({"id": card["chart_id"], "title": card["name"], "series": card["chart_payload"]})
        commodity_cards_html.append(
            f"""
            <div class="commodity-card chart-select-card" data-chart-id="{html.escape(card['chart_id'])}" data-compare-title="{html.escape(card['name'])}">
              <div class="commodity-top">
                <div>
                  <div class="market-symbol">{html.escape(card['symbol'])}</div>
                  <div class="market-name">{html.escape(card['name'])}</div>
                </div>
                <div class="market-desc">{html.escape(card['desc'])}</div>
              </div>
              <div class="commodity-value">{html.escape(card['value'])}</div>
              <div class="commodity-change {card['trend_class']}">{html.escape(card['change_pct'])}</div>
              <div class="line-chart chart-md" id="{html.escape(card['chart_id'])}"></div>
            </div>
            """
        )

    valuation_cards_html = []
    for card in valuation_cards:
        chart_instances.append({"id": card["chart_id"], "title": card["name"], "series": card["chart_payload"]})
        valuation_cards_html.append(
            f"""
            <div class="valuation-card chart-select-card" data-chart-id="{html.escape(card['chart_id'])}" data-compare-title="{html.escape(card['name'])}">
              <div class="valuation-top">
                <div>
                  <div class="market-symbol">{html.escape(card['symbol'])}</div>
                  <div class="market-name">{html.escape(card['name'])}</div>
                </div>
                <div class="market-desc">{html.escape(card['desc'])}</div>
              </div>
              <div class="valuation-metrics">
                <div class="valuation-metric">
                  <div class="valuation-label">当前 PE</div>
                  <div class="valuation-value">{html.escape(card['pe_value'])}</div>
                  <div class="valuation-change {card['trend_class']}">{html.escape(card['pe_change'])}</div>
                </div>
                <div class="valuation-metric">
                  <div class="valuation-label">当前分位</div>
                  <div class="valuation-value">{html.escape(card['percentile_value'])}</div>
                  <div class="valuation-change {card['percentile_class']}">{html.escape(card['percentile_change'])}</div>
                </div>
              </div>
              <div class="line-chart chart-md" id="{html.escape(card['chart_id'])}"></div>
            </div>
            """
        )

    category_sections_html = []
    for category in categories:
        rep = category["representative"]
        rep_html = ""
        if rep:
            rep_html = f"""
            <div class="category-highlight" style="background:{category['bg']};">
              <div class="highlight-badge" style="color:{category['color']}; border-color:{category['color']};">代表指标</div>
              <h3>{html.escape(rep['name'])}</h3>
              <div class="highlight-meta">
                <span>{html.escape(rep['latest_value'])}</span>
                <span class="{rep['trend_class']}">{html.escape(rep['latest_delta'])}</span>
                <span>{html.escape(rep['importance'])}</span>
              </div>
              <p>{html.escape(rep['reason'])}</p>
            </div>
            """

        cards_html = []
        for card in category["cards"]:
            chart_html = ""
            card_class = "indicator-card"
            card_attrs = ""
            if card["chart_payload"]:
                chart_instances.append({"id": card["chart_id"], "title": card["name"], "series": card["chart_payload"]})
                card_class = "indicator-card chart-select-card"
                card_attrs = f'data-chart-id="{html.escape(card["chart_id"])}" data-compare-title="{html.escape(card["name"])}"'
                chart_html = f'<div class="line-chart chart-lg" id="{html.escape(card["chart_id"])}"></div>'
            else:
                chart_html = '<div class="chart-placeholder">当前没有可用时序数据</div>'
            data_link = f'<a href="{html.escape(card["data"])}" target="_blank">CSV</a>' if card["data"] else ""
            note_line = card["note"] or card["error"] or "待补充数据源或抓取逻辑。"
            cards_html.append(
                f"""
                <div class="{card_class}" {card_attrs}>
                  <div class="indicator-header">
                    <div>
                      <div class="indicator-title-row">
                        <h4>{html.escape(card['name'])}</h4>
                        {'<span class="rep-tag">代表</span>' if card['is_representative'] else ''}
                      </div>
                      <div class="indicator-meta">
                        <span class="status {card['status']}">{html.escape(card['status'])}</span>
                        <span>重要性 {html.escape(card['importance'])}</span>
                        <span>公开性 {html.escape(card['publicity'])}</span>
                      </div>
                    </div>
                    <div class="indicator-right">
                      <div class="indicator-value">{html.escape(card['latest_value'])}</div>
                      <div class="indicator-delta {card['trend_class']}">{html.escape(card['latest_delta'])}</div>
                    </div>
                  </div>
                  <div class="mini-label">走势图（默认近 1 个月，可鼠标缩放/拖拽）</div>
                  {chart_html}
                  <p class="indicator-reason">{html.escape(card['reason'])}</p>
                  <div class="indicator-links">{data_link}</div>
                  <div class="indicator-detail-grid">
                    <div><strong>含义</strong><span>{html.escape(card['meaning'] or '待补充')}</span></div>
                    <div><strong>影响</strong><span>{html.escape(card['impact'] or '待补充')}</span></div>
                    <div><strong>参考值</strong><span>{html.escape(card['range'] or '待补充')}</span></div>
                    <div><strong>来源</strong><span>{html.escape(card['source'])}</span></div>
                  </div>
                  <p class="indicator-note">{html.escape(note_line)}</p>
                </div>
                """
            )

        category_sections_html.append(
            f"""
            <section class="category-section" id="cat-{html.escape(category['slug'])}">
              <div class="category-head">
                <div class="category-title-wrap">
                  <span class="category-icon" style="background:{category['bg']}; color:{category['color']};">{html.escape(category['icon'])}</span>
                  <div>
                    <h2>{html.escape(category['title'])}</h2>
                    <p>已接通 {category['fetched_count']} / {category['total_count']} 个指标，适合按主题监控宏观与市场结构变化。</p>
                  </div>
                </div>
                <div class="category-stat" style="color:{category['color']};">{category['fetched_count']}/{category['total_count']}</div>
              </div>
              {rep_html}
              <div class="indicator-grid">
                {''.join(cards_html)}
              </div>
            </section>
            """
        )

    nav_html = "".join(
        f'<a href="#{html.escape(anchor)}">{html.escape(label)}</a>' for anchor, label in nav_links
    )
    chart_instances_json = json.dumps(chart_instances, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>市场指标监控面板</title>
  <style>
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f3f6fb; color: #0f172a; overflow-x: hidden; }}
    a {{ color: inherit; text-decoration: none; }}
    .layout {{ display: flex; min-height: 100vh; }}
    .sidebar {{ width: 230px; background: linear-gradient(180deg, #08152f 0%, #0e2349 100%); color: #dbeafe; padding: 22px 18px; position: sticky; top: 0; height: 100vh; overflow: auto; }}
    .brand {{ display: flex; align-items: center; gap: 12px; margin-bottom: 28px; }}
    .brand-logo {{ width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-weight: 700; background: rgba(255,255,255,0.14); }}
    .brand-title {{ font-size: 16px; font-weight: 700; }}
    .brand-sub {{ font-size: 12px; color: #93c5fd; margin-top: 2px; }}
    .nav {{ display: flex; flex-direction: column; gap: 8px; }}
    .nav a {{ padding: 10px 12px; border-radius: 12px; color: #dbeafe; font-size: 14px; white-space: nowrap; }}
    .nav a:hover {{ background: rgba(255,255,255,0.08); }}
    .main {{ flex: 1; padding: 26px; }}
    .section, .category-section {{ scroll-margin-top: 84px; }}
    .hero {{ background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%); border: 1px solid #dbe4f0; border-radius: 24px; padding: 24px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06); }}
    .hero-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; }}
    .hero-title {{ font-size: 30px; font-weight: 800; margin: 0; }}
    .hero-desc {{ margin: 8px 0 0; color: #475569; line-height: 1.6; max-width: 780px; }}
    .hero-side {{ display: flex; flex-direction: column; align-items: flex-end; gap: 10px; }}
    .hero-actions-row {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .hero-tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .hero-tags span {{ font-size: 12px; border-radius: 999px; background: #e2e8f0; color: #334155; padding: 6px 10px; }}
    .refresh-btn {{ border: 0; background: #2563eb; color: #fff; border-radius: 12px; padding: 10px 16px; font-size: 14px; font-weight: 700; cursor: pointer; box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22); }}
    .refresh-btn:hover {{ background: #1d4ed8; }}
    .refresh-btn:disabled {{ background: #94a3b8; cursor: not-allowed; box-shadow: none; }}
    .secondary-btn {{ display: inline-flex; align-items: center; justify-content: center; border: 1px solid #bfdbfe; background: #eff6ff; color: #1d4ed8; border-radius: 12px; padding: 10px 16px; font-size: 14px; font-weight: 700; min-height: 42px; }}
    .secondary-btn:hover {{ background: #dbeafe; }}
    .refresh-status {{ min-height: 18px; font-size: 12px; color: #475569; text-align: right; }}
    .compare-panel {{ background: #fff; border: 1px solid #dbe4f0; border-radius: 24px; padding: 20px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06); }}
    .compare-toolbar {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }}
    .compare-title {{ font-size: 22px; font-weight: 800; margin: 0; }}
    .compare-subtitle {{ margin: 6px 0 0; color: #475569; line-height: 1.6; }}
    .compare-actions {{ display: flex; align-items: center; gap: 10px; }}
    .compare-count {{ font-size: 13px; color: #475569; }}
    .compare-btn {{ border: 1px solid #cbd5e1; background: #fff; color: #0f172a; border-radius: 10px; padding: 8px 12px; font-size: 13px; font-weight: 700; cursor: pointer; }}
    .compare-btn:hover {{ background: #f8fafc; }}
    .compare-btn:disabled {{ cursor: not-allowed; opacity: 0.45; }}
    .compare-selected {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .compare-chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: #eff6ff; color: #1d4ed8; font-size: 12px; font-weight: 700; }}
    .compare-chip button {{ border: 0; background: transparent; color: inherit; cursor: pointer; font-size: 14px; line-height: 1; padding: 0; }}
    .compare-empty {{ margin-top: 14px; padding: 18px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #64748b; background: #f8fafc; }}
    .compare-chart {{ height: 420px; margin-top: 16px; }}
    .market-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .market-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); display: block; }}
    .market-card-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .market-symbol {{ font-size: 12px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
    .market-name {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}
    .market-desc {{ max-width: 120px; font-size: 12px; color: #64748b; text-align: right; }}
    .market-value {{ font-size: 28px; font-weight: 800; margin-top: 12px; }}
    .market-change {{ margin-top: 6px; font-size: 13px; font-weight: 700; }}
    .up {{ color: #16a34a; }}
    .down {{ color: #dc2626; }}
    .flat {{ color: #64748b; }}
    .line-chart {{ width: 100%; background: #fff; border: 1px solid #dbe4f0; border-radius: 12px; }}
    .chart-sm {{ height: 140px; margin-top: 12px; }}
    .chart-md {{ height: 180px; margin-top: 12px; }}
    .chart-lg {{ height: 220px; margin-top: 10px; }}
    .section {{ margin-top: 24px; }}
    .section-title {{ font-size: 22px; font-weight: 800; margin: 0 0 14px; }}
    .signal-grid {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 12px; }}
    .signal-card {{ background: #fff; border-radius: 16px; border: 1px solid #e2e8f0; padding: 14px; }}
    .signal-label {{ font-size: 12px; color: #64748b; }}
    .signal-name {{ font-size: 14px; font-weight: 700; margin-top: 8px; min-height: 40px; }}
    .signal-value {{ font-size: 22px; font-weight: 800; margin-top: 10px; }}
    .signal-delta {{ font-size: 13px; font-weight: 700; margin-top: 6px; }}
    .commodity-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .commodity-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
    .commodity-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .commodity-value {{ font-size: 26px; font-weight: 800; margin-top: 10px; }}
    .commodity-change {{ margin-top: 6px; font-size: 13px; font-weight: 700; }}
    .valuation-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .valuation-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
    .valuation-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .valuation-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }}
    .valuation-metric {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px; }}
    .valuation-label {{ font-size: 12px; color: #64748b; }}
    .valuation-value {{ font-size: 24px; font-weight: 800; margin-top: 8px; }}
    .valuation-change {{ margin-top: 6px; font-size: 13px; font-weight: 700; }}
    .category-section {{ margin-top: 28px; background: #fff; border: 1px solid #e2e8f0; border-radius: 22px; padding: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
    .category-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 18px; }}
    .category-title-wrap {{ display: flex; align-items: center; gap: 14px; }}
    .category-title-wrap h2 {{ margin: 0; font-size: 22px; }}
    .category-title-wrap p {{ margin: 6px 0 0; color: #64748b; }}
    .category-icon {{ width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-weight: 800; }}
    .category-stat {{ font-size: 28px; font-weight: 800; }}
    .category-highlight {{ border-radius: 18px; padding: 18px; margin-bottom: 18px; }}
    .highlight-badge {{ display: inline-flex; align-items: center; padding: 5px 10px; border-radius: 999px; border: 1px solid currentColor; font-size: 12px; font-weight: 700; margin-bottom: 10px; }}
    .category-highlight h3 {{ margin: 0; font-size: 22px; }}
    .highlight-meta {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-weight: 700; }}
    .category-highlight p {{ margin: 12px 0 0; line-height: 1.6; color: #334155; }}
    .indicator-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .indicator-card {{ border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; background: #fbfdff; }}
    .indicator-header {{ display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }}
    .indicator-title-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .indicator-title-row h4 {{ margin: 0; font-size: 18px; }}
    .rep-tag {{ font-size: 11px; border-radius: 999px; background: #dbeafe; color: #1d4ed8; padding: 4px 8px; font-weight: 700; }}
    .indicator-meta {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: #64748b; }}
    .status {{ display: inline-flex; padding: 4px 8px; border-radius: 999px; font-weight: 700; }}
    .status.fetched {{ background: #dcfce7; color: #166534; }}
    .status.fetched_cached {{ background: #dbeafe; color: #1d4ed8; }}
    .status.fetch_failed {{ background: #fee2e2; color: #991b1b; }}
    .status.pending_manual {{ background: #fef3c7; color: #92400e; }}
    .status.skipped {{ background: #e0e7ff; color: #3730a3; }}
    .indicator-right {{ text-align: right; min-width: 110px; }}
    .indicator-value {{ font-size: 24px; font-weight: 800; }}
    .indicator-delta {{ margin-top: 6px; font-size: 13px; font-weight: 700; }}
    .mini-label {{ font-size: 12px; color: #64748b; margin-top: 10px; }}
    .indicator-reason {{ margin: 8px 0 12px; color: #334155; font-size: 14px; line-height: 1.6; }}
    .chart-placeholder {{ border: 1px dashed #cbd5e1; border-radius: 12px; padding: 24px; text-align: center; color: #64748b; background: #fff; }}
    .indicator-links {{ margin: 10px 0; }}
    .indicator-links a {{ color: #2563eb; font-size: 13px; font-weight: 700; }}
    .indicator-detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 14px; margin-top: 12px; }}
    .indicator-detail-grid div {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px; }}
    .indicator-detail-grid strong {{ display: block; font-size: 12px; color: #64748b; margin-bottom: 6px; }}
    .indicator-detail-grid span {{ font-size: 13px; color: #0f172a; line-height: 1.5; }}
    .indicator-note {{ font-size: 12px; color: #64748b; line-height: 1.6; margin: 12px 0 0; }}
    .chart-select-card {{ position: relative; cursor: pointer; transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease; }}
    .chart-select-card:hover {{ border-color: #93c5fd; box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08); }}
    .chart-select-card.selected {{ border-color: #2563eb; box-shadow: 0 14px 32px rgba(37, 99, 235, 0.16); }}
    .chart-select-card.selected::after {{ content: "已选中"; position: absolute; top: 12px; right: 12px; background: #2563eb; color: #fff; font-size: 11px; font-weight: 700; border-radius: 999px; padding: 4px 8px; }}
    @media (max-width: 1280px) {{
      .hero-title {{ font-size: 28px; }}
      .signal-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      .market-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .valuation-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .commodity-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 960px) {{
      .layout {{ display: block; }}
      .sidebar {{ width: auto; height: auto; position: sticky; top: 0; z-index: 30; padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
      .brand {{ margin-bottom: 12px; }}
      .brand-logo {{ width: 36px; height: 36px; border-radius: 10px; }}
      .nav {{ flex-direction: row; gap: 10px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: none; }}
      .nav::-webkit-scrollbar {{ display: none; }}
      .nav a {{ display: inline-flex; align-items: center; min-height: 40px; background: rgba(255,255,255,0.08); }}
      .main {{ padding: 18px; }}
      .hero {{ border-radius: 20px; }}
      .hero-top {{ flex-direction: column; align-items: stretch; }}
      .hero-side {{ align-items: stretch; }}
      .hero-actions-row {{ flex-direction: column; }}
      .refresh-btn {{ width: 100%; min-height: 44px; }}
      .secondary-btn {{ width: 100%; min-height: 44px; }}
      .refresh-status {{ text-align: left; }}
      .hero-tags {{ overflow-x: auto; flex-wrap: nowrap; padding-bottom: 2px; }}
      .hero-tags span {{ white-space: nowrap; }}
      .compare-toolbar {{ flex-direction: column; }}
      .compare-actions {{ width: 100%; justify-content: space-between; }}
      .compare-chart {{ height: 340px; }}
      .category-head {{ align-items: flex-start; flex-direction: column; }}
      .category-title-wrap {{ align-items: flex-start; }}
      .category-stat {{ font-size: 24px; }}
      .indicator-header {{ flex-direction: column; }}
      .indicator-right {{ text-align: left; min-width: 0; }}
      .signal-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .indicator-grid {{ grid-template-columns: 1fr; }}
      .market-grid {{ grid-template-columns: 1fr; }}
      .valuation-grid {{ grid-template-columns: 1fr; }}
      .commodity-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      .main {{ padding: 14px; }}
      .sidebar {{ padding: 12px 14px; }}
      .brand-title {{ font-size: 15px; }}
      .brand-sub {{ font-size: 11px; }}
      .hero {{ padding: 16px; border-radius: 18px; }}
      .hero-title {{ font-size: 24px; line-height: 1.25; }}
      .hero-desc, .compare-subtitle {{ font-size: 13px; }}
      .section {{ margin-top: 18px; }}
      .section-title, .compare-title, .category-title-wrap h2, .category-highlight h3 {{ font-size: 20px; }}
      .market-card, .commodity-card, .valuation-card, .indicator-card, .compare-panel, .category-section {{ padding: 14px; border-radius: 16px; }}
      .market-card-top, .commodity-top, .valuation-top {{ flex-direction: column; }}
      .market-desc {{ max-width: none; text-align: left; }}
      .market-value, .commodity-value, .valuation-value, .indicator-value, .signal-value {{ font-size: 22px; }}
      .chart-sm {{ height: 180px; }}
      .chart-md {{ height: 220px; }}
      .chart-lg {{ height: 260px; }}
      .compare-chart {{ height: 300px; }}
      .compare-actions {{ flex-direction: column; align-items: stretch; }}
      .compare-btn {{ min-height: 42px; }}
      .signal-grid {{ grid-template-columns: 1fr; }}
      .indicator-detail-grid {{ grid-template-columns: 1fr; }}
      .valuation-metrics {{ grid-template-columns: 1fr; }}
      .highlight-meta {{ gap: 8px; }}
      .chart-select-card.selected::after {{ top: 10px; right: 10px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-logo">MM</div>
        <div>
          <div class="brand-title">Market Monitor</div>
          <div class="brand-sub">大盘与分类指标监控</div>
        </div>
      </div>
      <nav class="nav">
        {nav_html}
      </nav>
    </aside>
    <main class="main">
      <section class="hero" id="overview">
        <div class="hero-top">
          <div>
            <h1 class="hero-title">市场指标监控面板</h1>
            <p class="hero-desc">一级页面聚焦大盘总览，二级页面按货币流动性、通胀、增长、估值、风险、技术结构和全球宏观分层展示。数据全部来自当前项目已有输出，适合直接做日常监控。</p>
          </div>
          <div class="hero-side">
            <div class="hero-actions-row">
              <button class="refresh-btn" id="refresh-data-btn" type="button">刷新数据</button>
              <a class="secondary-btn" href="strategy_dashboard.html">量化策略</a>
            </div>
            <div class="refresh-status" id="refresh-status">点击后重新拉取数据，失败项自动保留旧数据</div>
            <div class="hero-tags">
              <span>静态 HTML</span>
              <span>复用现有 CSV 数据</span>
              <span>总览 + 分类双层结构</span>
            </div>
          </div>
        </div>
        <div class="market-grid">
          {''.join(market_cards_html)}
        </div>
      </section>

      <section class="section" id="compare">
        <div class="compare-panel">
          <div class="compare-toolbar">
            <div>
              <h2 class="compare-title">图表对比</h2>
              <p class="compare-subtitle">点击任意图表卡片即可加入对比。系统会用每张卡片的主序列生成多坐标轴对比图，便于跨市场、跨指标同步观察。</p>
            </div>
            <div class="compare-actions">
              <span class="compare-count" id="compare-count">已选 0 项</span>
              <button class="compare-btn" id="clear-compare" type="button" disabled>清空选择</button>
            </div>
          </div>
          <div class="compare-selected" id="compare-selected"></div>
          <div class="compare-empty" id="compare-empty">当前未选择图表。点击页面中的任意图表卡片即可加入对比。</div>
          <div class="line-chart compare-chart" id="compare-chart"></div>
        </div>
      </section>

      <section class="section" id="macro">
        <h2 class="section-title">一级指标</h2>
        <div class="signal-grid">
          {''.join(macro_items_html)}
        </div>
      </section>

      <section class="section" id="valuations">
        <h2 class="section-title">PE 估值分位</h2>
        <p class="hero-desc">统一展示标普500、纳斯达克和沪深300 的 PE 与 PE 分位。分位定义为历史上 PE 低于当日 PE 的天数占比。</p>
        <div class="valuation-grid">
          {''.join(valuation_cards_html)}
        </div>
      </section>

      <section class="section" id="commodity-trends">
        <h2 class="section-title">大宗商品价格走势</h2>
        <p class="hero-desc">单独监控商品相关宏观价格走势，覆盖综合商品指数、能源、贵金属、工业金属、农产品和工业链代理指标。</p>
        <div class="commodity-grid">
          {''.join(commodity_cards_html)}
        </div>
      </section>

      <section class="section" id="categories">
        <h2 class="section-title">二级分类监控</h2>
        {''.join(category_sections_html)}
      </section>
    </main>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <script>
    const chartInstances = {chart_instances_json};
    const chartMetaMap = new Map(chartInstances.map((item) => [item.id, item]));
    const isCompactScreen = () => window.innerWidth <= 640;
    const isTabletScreen = () => window.innerWidth <= 960;

    function renderCharts() {{
      const renderedCharts = [];
      const sharedGroup = "market-monitor-sync";

      chartInstances.forEach((item) => {{
        const el = document.getElementById(item.id);
        if (!el || !item.series || item.series.length === 0) return;

        const chart = echarts.init(el);
        chart.group = sharedGroup;
        const allDates = item.series[0].x || [];
        const defaultStartDate = (() => {{
          if (allDates.length === 0) return 0;
          const endDate = new Date(allDates[allDates.length - 1]);
          const startDate = new Date(endDate.getTime());
          startDate.setMonth(startDate.getMonth() - 1);
          const idx = allDates.findIndex((d) => new Date(d) >= startDate);
          return idx <= 0 ? 0 : (idx / Math.max(allDates.length - 1, 1)) * 100;
        }})();

        const compact = isCompactScreen();
        const tablet = isTabletScreen();
        const option = {{
          animation: false,
          tooltip: {{ trigger: "axis" }},
          legend: {{
            show: item.series.length > 1 && !compact,
            top: 2,
            right: 6,
            textStyle: {{ fontSize: compact ? 10 : 11 }},
          }},
          grid: {{
            top: item.series.length > 1 && !compact ? 30 : 18,
            left: compact ? 40 : 48,
            right: compact ? 10 : 18,
            bottom: compact ? 42 : 48,
          }},
          xAxis: {{
            type: "category",
            boundaryGap: false,
            axisLabel: {{ color: "#64748b", fontSize: compact ? 10 : 11, hideOverlap: true }},
            data: allDates,
          }},
          yAxis: {{
            type: "value",
            scale: true,
            axisLabel: {{ color: "#64748b", fontSize: compact ? 10 : 11 }},
            splitLine: {{ lineStyle: {{ color: "#e2e8f0" }} }},
          }},
          dataZoom: [
            {{
              type: "inside",
              zoomOnMouseWheel: true,
              moveOnMouseMove: true,
              moveOnMouseWheel: true,
              start: defaultStartDate,
              end: 100,
            }},
            {{
              type: "slider",
              start: defaultStartDate,
              end: 100,
              height: compact ? 14 : 18,
              bottom: compact ? 6 : 8,
              showDetail: !tablet,
            }},
          ],
          series: item.series.map((s, index) => ({{
            name: s.name,
            type: "line",
            showSymbol: false,
            smooth: false,
            lineStyle: {{ width: 2 }},
            areaStyle: item.series.length === 1 ? {{ opacity: 0.08 }} : undefined,
            data: s.y,
          }})),
        }};

        chart.setOption(option);
        renderedCharts.push(chart);
        window.addEventListener("resize", () => chart.resize());
      }});

      if (renderedCharts.length > 1) {{
        echarts.connect(sharedGroup);
      }}
    }}

    function buildCompareOption(selectedItems) {{
      const dateSet = new Set();
      selectedItems.forEach((item) => {{
        const firstSeries = item.series[0];
        (firstSeries?.x || []).forEach((value) => dateSet.add(value));
      }});
      const dates = Array.from(dateSet).sort();
      const colorPalette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0f766e", "#ea580c", "#0891b2"];
      const yAxes = [];
      const series = [];
      let leftAxisCount = 0;
      let rightAxisCount = 0;
      const compact = isCompactScreen();

      selectedItems.forEach((item, index) => {{
        const firstSeries = item.series[0];
        if (!firstSeries) return;
        const color = colorPalette[index % colorPalette.length];
        const side = index % 2 === 0 ? "left" : "right";
        const offset = side === "left" ? leftAxisCount * (compact ? 34 : 56) : rightAxisCount * (compact ? 34 : 56);
        if (side === "left") {{
          leftAxisCount += 1;
        }} else {{
          rightAxisCount += 1;
        }}
        const valueMap = new Map(firstSeries.x.map((xValue, idx) => [xValue, firstSeries.y[idx]]));
        yAxes.push({{
          type: "value",
          name: compact ? "" : (item.title || firstSeries.name),
          nameTextStyle: {{ color }},
          axisLine: {{ show: true, lineStyle: {{ color }} }},
          axisLabel: {{ color, fontSize: compact ? 10 : 11 }},
          splitLine: {{ show: index === 0, lineStyle: {{ color: "#e2e8f0" }} }},
          position: side,
          offset,
          scale: true,
        }});
        series.push({{
          name: item.title || firstSeries.name,
          type: "line",
          yAxisIndex: yAxes.length - 1,
          showSymbol: false,
          smooth: false,
          lineStyle: {{ width: 2, color }},
          itemStyle: {{ color }},
          emphasis: {{ focus: "series" }},
          data: dates.map((date) => valueMap.has(date) ? valueMap.get(date) : null),
        }});
      }});

      const gridLeft = (compact ? 48 : 70) + Math.max(0, leftAxisCount - 1) * (compact ? 34 : 56);
      const gridRight = (compact ? 48 : 70) + Math.max(0, rightAxisCount - 1) * (compact ? 34 : 56);

      return {{
        animation: false,
        color: colorPalette,
        tooltip: {{ trigger: "axis" }},
        legend: {{ top: 6, textStyle: {{ fontSize: compact ? 10 : 11 }} }},
        grid: {{ top: compact ? 44 : 54, left: gridLeft, right: gridRight, bottom: compact ? 48 : 60 }},
        xAxis: {{
          type: "category",
          boundaryGap: false,
          data: dates,
          axisLabel: {{ color: "#64748b", fontSize: compact ? 10 : 11, hideOverlap: true }},
        }},
        yAxis: yAxes,
        dataZoom: [
          {{ type: "inside", start: 0, end: 100 }},
          {{ type: "slider", start: 0, end: 100, height: compact ? 14 : 18, bottom: compact ? 8 : 10, showDetail: !compact }},
        ],
        series,
      }};
    }}

    function setupCompareSelection() {{
      const selectedIds = new Set();
      const chartEl = document.getElementById("compare-chart");
      const chipEl = document.getElementById("compare-selected");
      const emptyEl = document.getElementById("compare-empty");
      const countEl = document.getElementById("compare-count");
      const clearBtn = document.getElementById("clear-compare");
      const cardEls = Array.from(document.querySelectorAll(".chart-select-card[data-chart-id]"));
      const compareChart = echarts.init(chartEl);

      function syncCardState() {{
        cardEls.forEach((cardEl) => {{
          const isSelected = selectedIds.has(cardEl.dataset.chartId);
          cardEl.classList.toggle("selected", isSelected);
        }});
      }}

      function renderSelectedChips() {{
        const selectedCards = cardEls.filter((cardEl) => selectedIds.has(cardEl.dataset.chartId));
        chipEl.innerHTML = selectedCards.map((cardEl) => `
          <span class="compare-chip">
            ${{cardEl.dataset.compareTitle}}
            <button type="button" data-remove-id="${{cardEl.dataset.chartId}}" aria-label="移除">×</button>
          </span>
        `).join("");
      }}

      function renderCompareChart() {{
        const selectedItems = Array.from(selectedIds)
          .map((id) => chartMetaMap.get(id))
          .filter((item) => item && item.series && item.series.length > 0);
        countEl.textContent = `已选 ${{selectedItems.length}} 项`;
        clearBtn.disabled = selectedItems.length === 0;
        emptyEl.style.display = selectedItems.length === 0 ? "block" : "none";
        chartEl.style.display = selectedItems.length === 0 ? "none" : "block";
        if (selectedItems.length === 0) {{
          compareChart.clear();
          return;
        }}
        compareChart.setOption(buildCompareOption(selectedItems), true);
      }}

      cardEls.forEach((cardEl) => {{
        cardEl.addEventListener("click", (event) => {{
          if (event.target.closest("a") || event.target.closest("button")) return;
          const chartId = cardEl.dataset.chartId;
          if (selectedIds.has(chartId)) {{
            selectedIds.delete(chartId);
          }} else {{
            selectedIds.add(chartId);
          }}
          syncCardState();
          renderSelectedChips();
          renderCompareChart();
        }});
      }});

      chipEl.addEventListener("click", (event) => {{
        const button = event.target.closest("button[data-remove-id]");
        if (!button) return;
        selectedIds.delete(button.dataset.removeId);
        syncCardState();
        renderSelectedChips();
        renderCompareChart();
      }});

      clearBtn.addEventListener("click", () => {{
        selectedIds.clear();
        syncCardState();
        renderSelectedChips();
        renderCompareChart();
      }});

      renderSelectedChips();
      renderCompareChart();
      window.addEventListener("resize", () => compareChart.resize());
    }}

    async function refreshDashboardData() {{
      const refreshBtn = document.getElementById("refresh-data-btn");
      const statusEl = document.getElementById("refresh-status");
      if (!refreshBtn || !statusEl) return;

      refreshBtn.disabled = true;
      refreshBtn.textContent = "刷新中...";
      statusEl.textContent = "正在拉取最新数据并重建页面，请稍候。";

      try {{
        const response = await fetch("/api/refresh", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
        }});
        const responseText = await response.text();
        const contentType = (response.headers.get("content-type") || "").toLowerCase();
        if (!contentType.includes("application/json")) {{
          throw new Error("刷新接口返回了 HTML 页面，请使用 `python3 dashboard_server.py` 打开页面，或检查 `/api/refresh` 的反向代理配置。");
        }}
        const payload = JSON.parse(responseText);
        if (!response.ok || !payload.ok) {{
          throw new Error(payload.message || "刷新失败");
        }}
        statusEl.textContent = `刷新完成，耗时 ${{payload.duration_seconds || "--"}} 秒，页面即将更新。`;
        window.setTimeout(() => {{
          const nextUrl = new URL(window.location.href);
          nextUrl.searchParams.set("ts", String(Date.now()));
          window.location.href = nextUrl.toString();
        }}, 800);
      }} catch (error) {{
        statusEl.textContent = `刷新失败：${{error.message || error}}`;
        refreshBtn.disabled = false;
        refreshBtn.textContent = "刷新数据";
      }}
    }}

    function setupRefreshButton() {{
      const refreshBtn = document.getElementById("refresh-data-btn");
      if (!refreshBtn) return;
      refreshBtn.addEventListener("click", () => {{
        refreshDashboardData();
      }});
    }}

    renderCharts();
    setupCompareSelection();
    setupRefreshButton();
  </script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_HTML.write_text(render_dashboard(), encoding="utf-8")
    print(f"Generated: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
