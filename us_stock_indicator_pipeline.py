from __future__ import annotations

import argparse
import ast
import html
import hashlib
from io import StringIO
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


LOOKBACK_YEARS = 5
BASE_DIR = Path(__file__).resolve().parent
GUIDE_PATH = BASE_DIR / "us_stock_market_indicators_guide.md"
OUTPUT_DIR = BASE_DIR / "output" / "us_stock_indicators"
EXISTING_OUTPUT_DIR = BASE_DIR / "output"
TIMEOUT = 20
DEBUG_ENV_FILE = BASE_DIR / ".dbg" / "indicator-fetch-failures.env"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


CATEGORY_SLUG_MAP = {
    "货币流动性": "money_liquidity",
    "通胀": "inflation",
    "经济增长 / 基本面": "growth_fundamentals",
    "企业盈利与估值": "earnings_valuation",
    "风险偏好与压力指标": "risk_stress",
    "资金流向与仓位": "flows_positioning",
    "技术面与市场结构": "technical_structure",
    "市场情绪与调查": "sentiment_surveys",
    "全球宏观与地缘": "global_macro",
}


plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Heiti SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class IndicatorConfig:
    slug: str
    fetcher: Callable[[], pd.DataFrame] | None
    public_source: str
    note: str = ""
    core: bool = False
    proxy: bool = False


def stable_slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if normalized:
        return normalized
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"indicator-{digest}"


def category_slug(title: str) -> str:
    return CATEGORY_SLUG_MAP.get(title, stable_slug(title))


def fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def load_fred_api_key() -> str | None:
    env_key = os.getenv("FRED_API_KEY", "").strip()
    if env_key:
        return env_key

    reference_file = BASE_DIR / "test_m.py"
    if not reference_file.exists():
        return None

    content = reference_file.read_text(encoding="utf-8")
    match = re.search(r'^FRED_API_KEY\s*=\s*["\']([^"\']+)["\']', content, re.M)
    if not match:
        return None

    file_key = match.group(1).strip()
    if not file_key or file_key == "YOUR_FRED_API_KEY_HERE":
        return None
    return file_key


def report_debug_event(hypothesis_id: str, location: str, msg: str, data: dict | None = None, run_id: str = "pre-fix") -> None:
    # #region debug-point A:report-event
    event_url = "http://127.0.0.1:7777/event"
    session_id = "indicator-fetch-failures"
    try:
        if DEBUG_ENV_FILE.exists():
            content = DEBUG_ENV_FILE.read_text(encoding="utf-8")
            for line in content.splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    event_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip()
        payload = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
        }
        req = urllib.request.Request(
            event_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=2).read()
    except Exception:
        pass
    # #endregion


def ensure_date_sorted(
    df: pd.DataFrame,
    *,
    years: int | None = LOOKBACK_YEARS,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date
    numeric_columns = [col for col in result.columns if col != "date"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=numeric_columns, how="all")
    result = result.sort_values("date").reset_index(drop=True)
    cutoff = None
    if start_date is not None:
        cutoff = pd.Timestamp(start_date).normalize()
    elif years is not None:
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    if cutoff is not None:
        result = result[pd.to_datetime(result["date"]) >= cutoff].reset_index(drop=True)
    return result


def fetch_fred_series(series_id: str, column_name: str = "value") -> pd.DataFrame:
    # #region debug-point A:fred-fetch
    url = fred_csv_url(series_id)
    report_debug_event("A", "fetch_fred_series", "start fred fetch", {"series_id": series_id, "url": url})
    try:
        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"},
        )
        response.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        columns = {column.upper(): column for column in df.columns}
        date_column = columns.get("DATE")
        value_column = columns.get(series_id.upper())
        if not date_column or not value_column:
            report_debug_event("A", "fetch_fred_series", "fred parse failed", {"series_id": series_id, "columns": list(df.columns)})
            raise ValueError(f"Failed to parse FRED series: {series_id}")
        df = df.rename(columns={date_column: "date", value_column: column_name})
        result = ensure_date_sorted(df[["date", column_name]])
        report_debug_event("A", "fetch_fred_series", "fred fetch ok", {"series_id": series_id, "rows": len(result)})
        return result
    except Exception as exc:
        report_debug_event("A", "fetch_fred_series", "fred fetch failed", {"series_id": series_id, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    # #endregion


def fetch_fred_series_full_history(
    series_id: str,
    column_name: str = "value",
    *,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    api_key = load_fred_api_key()
    if api_key:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        if start_date is not None:
            params["observation_start"] = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        api_response = None
        api_last_error: Exception | None = None
        for _ in range(3):
            try:
                api_response = requests.get(
                    FRED_OBSERVATIONS_URL,
                    params=params,
                    timeout=max(TIMEOUT, 30),
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                )
                api_response.raise_for_status()
                payload = api_response.json()
                observations = payload.get("observations")
                if observations is None:
                    raise ValueError(f"FRED API missing observations for {series_id}")
                rows = [
                    {"date": item["date"], column_name: float(item["value"])}
                    for item in observations
                    if item.get("value") not in {None, "."}
                ]
                if rows:
                    return ensure_date_sorted(pd.DataFrame(rows), years=None, start_date=start_date)
            except Exception as exc:  # noqa: PERF203
                api_last_error = exc
        report_debug_event(
            "A",
            "fetch_fred_series_full_history",
            "fred full-history api fallback to csv",
            {"series_id": series_id, "error_type": type(api_last_error).__name__ if api_last_error else "", "error": str(api_last_error) if api_last_error else ""},
        )

    url = fred_csv_url(series_id)
    response = None
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = requests.get(
                url,
                timeout=max(TIMEOUT, 30),
                headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"},
            )
            response.raise_for_status()
            break
        except Exception as exc:  # noqa: PERF203
            last_error = exc
    if response is None:
        raise ValueError(f"FRED full-history fetch failed for {series_id}: {last_error}")
    df = pd.read_csv(pd.io.common.StringIO(response.text))
    columns = {column.upper(): column for column in df.columns}
    date_column = columns.get("DATE") or columns.get("OBSERVATION_DATE")
    value_column = columns.get(series_id.upper())
    if not date_column or not value_column:
        raise ValueError(f"Failed to parse FRED series: {series_id}")
    df = df.rename(columns={date_column: "date", value_column: column_name})
    return ensure_date_sorted(df[["date", column_name]], years=None, start_date=start_date)


def fetch_fred_yoy(series_id: str, column_name: str) -> pd.DataFrame:
    df = fetch_fred_api_series(series_id, "level")
    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(df["level"], errors="coerce").pct_change(12) * 100
    return ensure_date_sorted(df[["date", column_name]])


def fetch_fred_qoq_annualized(series_id: str, column_name: str) -> pd.DataFrame:
    df = fetch_fred_api_series(series_id, "level")
    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = (df["level"] / df["level"].shift(1)) ** 4 - 1
    df[column_name] = df[column_name] * 100
    return ensure_date_sorted(df[["date", column_name]])


def fetch_fred_mom(series_id: str, column_name: str) -> pd.DataFrame:
    df = fetch_fred_api_series(series_id, "level")
    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(df["level"], errors="coerce").pct_change() * 100
    return ensure_date_sorted(df[["date", column_name]])


def fetch_fred_multi(series_map: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series_id, column_name in series_map.items():
        frames.append(fetch_fred_series(series_id, column_name))
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="date", how="outer")
    return ensure_date_sorted(merged)


def fetch_fred_api_series(
    series_id: str,
    column_name: str,
    *,
    frequency: str | None = None,
    aggregation_method: str | None = None,
) -> pd.DataFrame:
    api_key = load_fred_api_key()
    if api_key:
        start = (pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        if frequency:
            params["frequency"] = frequency
        if aggregation_method:
            params["aggregation_method"] = aggregation_method
        try:
            response = requests.get(
                FRED_OBSERVATIONS_URL,
                params=params,
                timeout=TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations")
            if observations is None:
                raise ValueError(f"FRED API missing observations for {series_id}")
            rows = [
                {"date": item["date"], column_name: float(item["value"])}
                for item in observations
                if item.get("value") not in {None, "."}
            ]
            if rows:
                return ensure_date_sorted(pd.DataFrame(rows))
        except Exception as exc:
            report_debug_event(
                "A",
                "fetch_fred_api_series",
                "fred observations fallback to csv",
                {"series_id": series_id, "error_type": type(exc).__name__, "error": str(exc)},
            )

    return fetch_fred_series(series_id, column_name)


def fetch_fred_rate_series(series_id: str, column_name: str) -> pd.DataFrame:
    return fetch_fred_api_series(
        series_id,
        column_name,
        frequency="d",
        aggregation_method="avg",
    )


def fetch_yfinance_close(ticker: str, column_name: str = "value") -> pd.DataFrame:
    # #region debug-point B:yfinance-close
    report_debug_event("B", "fetch_yfinance_close", "start yfinance close fetch", {"ticker": ticker})
    try:
        history = yf.Ticker(ticker).history(period=f"{LOOKBACK_YEARS}y", auto_adjust=False)
        if history.empty:
            report_debug_event("B", "fetch_yfinance_close", "empty yfinance response", {"ticker": ticker})
            raise ValueError(f"No price data returned for {ticker}")
        price_column = "Adj Close" if "Adj Close" in history.columns else "Close"
        df = history[[price_column]].copy()
        df = df.rename(columns={price_column: column_name}).reset_index()
        date_column = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_column: "date"})
        result = ensure_date_sorted(df[["date", column_name]])
        report_debug_event("B", "fetch_yfinance_close", "yfinance close ok", {"ticker": ticker, "rows": len(result)})
        return result
    except Exception as exc:
        report_debug_event("B", "fetch_yfinance_close", "yfinance close failed", {"ticker": ticker, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    # #endregion


def fetch_yfinance_ohlcv(ticker: str) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(period=f"{LOOKBACK_YEARS}y", auto_adjust=False)
    if history.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}")
    columns = [column for column in ["Close", "Volume"] if column in history.columns]
    df = history[columns].copy().reset_index()
    date_column = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_column: "date", "Close": "close", "Volume": "volume"})
    return ensure_date_sorted(df)


def fetch_existing_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    lowered = {column.lower(): column for column in df.columns}
    if "date" not in lowered:
        raise ValueError(f"CSV missing date column: {csv_path}")
    date_column = lowered["date"]
    value_columns = [column for column in df.columns if column != date_column]
    normalized = df[[date_column, *value_columns]].copy()
    normalized = normalized.rename(columns={date_column: "date"})
    return ensure_date_sorted(normalized)


def rename_single_value_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    value_columns = [column for column in df.columns if column != "date"]
    if len(value_columns) != 1:
        raise ValueError(f"Expected single value column, got {value_columns}")
    return df.rename(columns={value_columns[0]: column_name})[["date", column_name]]


def to_reference_value_frame(df: pd.DataFrame, date_col: str, value_col: str, column_name: str) -> pd.DataFrame:
    result = df[[date_col, value_col]].copy()
    result.columns = ["date", column_name]
    return ensure_date_sorted(result)


def fetch_public_csv(csv_url: str, date_column: str, value_columns: dict[str, str]) -> pd.DataFrame:
    response = requests.get(csv_url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(response.text))
    selected = df[[date_column, *value_columns.keys()]].copy()
    selected = selected.rename(columns={date_column: "date", **value_columns})
    return ensure_date_sorted(selected)


def fetch_crb_recent_series() -> pd.DataFrame:
    existing_csv = EXISTING_OUTPUT_DIR / "commodity_macro" / "crb_index_recent.csv"
    if existing_csv.exists():
        return fetch_existing_csv(existing_csv)

    url = "https://ru.stockq.org/index/CRB.php"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 12:
        raise ValueError("CRB history table not found")

    table_text = " ".join(tables[11].get_text(" ", strip=True).split())
    pattern = re.compile(r"(\d{4}/\d{2}/\d{2})\s+([\d.]+)\s+([+-]?[\d.]+)\s+%")
    rows = pattern.findall(table_text)
    if not rows:
        raise ValueError("CRB series parse failed")

    df = pd.DataFrame(rows, columns=["date", "value", "change_pct"])
    df = df[["date", "value"]]
    return ensure_date_sorted(df)


def fetch_commodity_macro_series(csv_name: str, column_name: str, symbol: str) -> pd.DataFrame:
    existing_csv = EXISTING_OUTPUT_DIR / "commodity_macro" / csv_name
    if existing_csv.exists():
        return rename_single_value_column(fetch_existing_csv(existing_csv), column_name)

    raw_df = ak.futures_foreign_hist(symbol=symbol)
    return to_reference_value_frame(raw_df, "date", "close", column_name)


def fetch_m2_yoy() -> pd.DataFrame:
    return fetch_fred_yoy("M2SL", "m2_yoy")


def fetch_pce_yoy() -> pd.DataFrame:
    return fetch_fred_yoy("PCEPILFE", "core_pce_yoy")


def fetch_cpi_yoy() -> pd.DataFrame:
    headline = fetch_fred_yoy("CPIAUCSL", "cpi_yoy")
    core = fetch_fred_mom("CPILFESL", "core_cpi_mom")
    merged = pd.merge(headline, core, on="date", how="outer")
    return ensure_date_sorted(merged)


def fetch_ppi_yoy() -> pd.DataFrame:
    return fetch_fred_mom("PPIACO", "ppi_change")


def fetch_oer_yoy() -> pd.DataFrame:
    return fetch_fred_yoy("CUSR0000SEHC", "oer_yoy")


def fetch_gdp_growth() -> pd.DataFrame:
    real_gdp = fetch_fred_qoq_annualized("GDPC1", "gdp_annualized")
    nominal_gdp = fetch_fred_qoq_annualized("GDP", "us_gdp_yoy_proxy")
    merged = pd.merge(real_gdp, nominal_gdp, on="date", how="outer")
    return ensure_date_sorted(merged)


def fetch_nfp_unemployment() -> pd.DataFrame:
    payrolls = fetch_fred_api_series("PAYEMS", "payems_level")
    payrolls["date"] = pd.to_datetime(payrolls["date"])
    payrolls["nfp_change_k"] = payrolls["payems_level"].diff() / 10
    payrolls = ensure_date_sorted(payrolls[["date", "nfp_change_k"]])
    unemployment = fetch_fred_api_series("UNRATE", "unemployment_rate")
    merged = pd.merge(payrolls, unemployment, on="date", how="outer")
    return ensure_date_sorted(merged)


def fetch_retail_sales() -> pd.DataFrame:
    return fetch_fred_mom("RSAFS", "retail_sales_mom")


def fetch_us_leading_proxy() -> pd.DataFrame:
    return fetch_ak_macro_series("macro_usa_cb_consumer_confidence", "日期", "今值", "consumer_confidence_proxy")


def fetch_ism_manufacturing() -> pd.DataFrame:
    try:
        return fetch_fred_api_series("NAPM", "ism_manufacturing_pmi")
    except Exception:
        return fetch_ak_macro_series("macro_usa_ism_pmi", "日期", "今值", "ism_manufacturing_pmi")


def fetch_ism_services() -> pd.DataFrame:
    return fetch_ak_macro_series("macro_usa_ism_non_pmi", "日期", "今值", "ism_services_pmi")


def fetch_fed_funds_rate() -> pd.DataFrame:
    try:
        return fetch_fred_rate_series("FEDFUNDS", "fed_funds_rate")
    except Exception as exc:
        report_debug_event(
            "A",
            "fetch_fed_funds_rate",
            "FEDFUNDS fallback to DFF monthly mean",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        daily = fetch_fred_rate_series("DFF", "fed_funds_rate")
        daily["date"] = pd.to_datetime(daily["date"])
        monthly = (
            daily.set_index("date")
            .resample("ME")
            .mean(numeric_only=True)
            .reset_index()
        )
        return ensure_date_sorted(monthly[["date", "fed_funds_rate"]])


def fetch_yield_curve_fred() -> pd.DataFrame:
    rates = pd.merge(
        fetch_fred_rate_series("DGS2", "us_2y"),
        fetch_fred_rate_series("DGS10", "us_10y"),
        on="date",
        how="outer",
    )
    rates["yield_curve_2y10y"] = rates["us_10y"] - rates["us_2y"]
    return ensure_date_sorted(rates[["date", "us_2y", "us_10y", "yield_curve_2y10y"]])


def fetch_real_rate_tips() -> pd.DataFrame:
    return fetch_fred_rate_series("DFII10", "real_rate_10y_tips")


def fetch_yield_curve_ak() -> pd.DataFrame:
    return fetch_ak_us_rate_series()[["date", "us_2y", "us_10y", "yield_curve_2y10y"]]


def fetch_real_rate_proxy() -> pd.DataFrame:
    rates = fetch_ak_us_rate_series()[["date", "us_10y"]]
    core_pce = fetch_pce_yoy()
    merged = pd.merge(rates, core_pce, on="date", how="inner")
    merged["real_rate_proxy"] = merged["us_10y"] - merged["core_pce_yoy"]
    return ensure_date_sorted(merged[["date", "real_rate_proxy", "us_10y", "core_pce_yoy"]])


def fetch_sp500_trend_from_existing() -> pd.DataFrame:
    price = fetch_ak_sp500_close()
    price["date"] = pd.to_datetime(price["date"])
    price["ma50"] = price["sp500_close"].rolling(50).mean()
    price["ma200"] = price["sp500_close"].rolling(200).mean()
    price["price_vs_ma200_pct"] = (price["sp500_close"] / price["ma200"] - 1) * 100
    return ensure_date_sorted(price[["date", "sp500_close", "ma50", "ma200", "price_vs_ma200_pct"]])


def fetch_vix_from_github() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"DATE": "date", "CLOSE": "vix", "OPEN": "open", "HIGH": "high", "LOW": "low"})
    return ensure_date_sorted(df[["date", "vix", "open", "high", "low"]])


def fetch_sector_rotation_ak() -> pd.DataFrame:
    tickers = {
        "SPY": "sp500",
        "XLK": "technology",
        "XLY": "consumer_discretionary",
        "XLU": "utilities",
    }
    merged = None
    for ticker, column in tickers.items():
        frame = fetch_ak_us_stock_daily(ticker, close_name=column)
        merged = frame if merged is None else pd.merge(merged, frame, on="date", how="inner")
    merged["xlk_vs_spy"] = merged["technology"] / merged["sp500"] * 100
    merged["xly_vs_spy"] = merged["consumer_discretionary"] / merged["sp500"] * 100
    merged["xlu_vs_spy"] = merged["utilities"] / merged["sp500"] * 100
    return ensure_date_sorted(merged[["date", "xlk_vs_spy", "xly_vs_spy", "xlu_vs_spy"]])


def fetch_volume_price_ak() -> pd.DataFrame:
    df = fetch_ak_us_stock_daily("SPY", close_name="close", include_volume=True)
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    return ensure_date_sorted(df[["date", "close", "volume", "volume_ma20"]])


def fetch_multpl_series(url: str, value_name: str) -> pd.DataFrame:
    # #region debug-point D:multpl
    report_debug_event("D", "fetch_multpl_series", "start multpl fetch", {"url": url, "value_name": value_name})
    try:
        response = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        match = re.search(r"let pi = (\[\[.*?\]\]);", response.text, re.S)
        if not match:
            report_debug_event("D", "fetch_multpl_series", "multpl parse failed", {"url": url})
            raise ValueError(f"Could not parse Multpl payload: {url}")
        payload_text = match.group(1)
        payload_text = re.sub(r"\bnull\b", "None", payload_text)
        payload_text = re.sub(r"\bNaN\b", "None", payload_text)
        payload_text = re.sub(r"\bInfinity\b", "None", payload_text)
        payload_text = re.sub(r"\bundefined\b", "None", payload_text)
        payload = ast.literal_eval(payload_text)
        offsets = payload[0]
        values = payload[1]
        df = pd.DataFrame({"offset": offsets, value_name: values})
        df["date"] = pd.Timestamp("1970-01-01") + pd.to_timedelta(df["offset"], unit="D")
        result = ensure_date_sorted(df[["date", value_name]])
        report_debug_event("D", "fetch_multpl_series", "multpl fetch ok", {"url": url, "rows": len(result)})
        return result
    except Exception as exc:
        report_debug_event("D", "fetch_multpl_series", "multpl fetch failed", {"url": url, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    # #endregion


def fetch_ak_macro_series(func_name: str, date_col: str, value_col: str, column_name: str) -> pd.DataFrame:
    raw_df = getattr(ak, func_name)()
    df = raw_df[[date_col, value_col]].copy()
    df.columns = ["date", column_name]
    return ensure_date_sorted(df)


def fetch_ak_us_rate_series() -> pd.DataFrame:
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)).strftime("%Y%m%d")
    raw_df = ak.bond_zh_us_rate(start_date=start_date)
    df = raw_df[
        [
            "日期",
            "美国国债收益率2年",
            "美国国债收益率10年",
            "美国国债收益率10年-2年",
            "美国GDP年增率",
        ]
    ].copy()
    df.columns = ["date", "us_2y", "us_10y", "yield_curve_2y10y", "us_gdp_yoy_proxy"]
    return ensure_date_sorted(df)


def fetch_ak_sp500_close() -> pd.DataFrame:
    existing_csv = EXISTING_OUTPUT_DIR / "sp500_daily.csv"
    if existing_csv.exists():
        df = fetch_existing_csv(existing_csv)
        numeric = [column for column in df.columns if column != "date"]
        if len(numeric) == 1:
            return df.rename(columns={numeric[0]: "sp500_close"})

    raw_df = ak.index_us_stock_sina(symbol=".INX")
    df = raw_df[["date", "close"]].copy()
    df.columns = ["date", "sp500_close"]
    return ensure_date_sorted(df)


def fetch_ak_us_stock_daily(symbol: str, close_name: str = "close", include_volume: bool = False) -> pd.DataFrame:
    # #region debug-point C:ak-us-stock
    report_debug_event("C", "fetch_ak_us_stock_daily", "start ak us stock fetch", {"symbol": symbol, "include_volume": include_volume})
    try:
        df = ak.stock_us_daily(symbol=symbol)
        columns = ["date", "close"]
        if include_volume:
            columns.append("volume")
        result = df[columns].copy()
        rename_map = {"close": close_name}
        if include_volume:
            rename_map["volume"] = "volume"
        result = result.rename(columns=rename_map)
        normalized = ensure_date_sorted(result)
        report_debug_event("C", "fetch_ak_us_stock_daily", "ak us stock ok", {"symbol": symbol, "rows": len(normalized)})
        return normalized
    except Exception as exc:
        report_debug_event("C", "fetch_ak_us_stock_daily", "ak us stock failed", {"symbol": symbol, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    # #endregion


def fetch_sp500_earnings() -> pd.DataFrame:
    return fetch_multpl_series("https://www.multpl.com/s-p-500-earnings", "sp500_eps_actual")


def fetch_sp500_pe_proxy() -> pd.DataFrame:
    existing_csv = EXISTING_OUTPUT_DIR / "us_pe_spy_daily.csv"
    if existing_csv.exists():
        df = fetch_existing_csv(existing_csv)
        numeric = [column for column in df.columns if column != "date"]
        if len(numeric) == 1:
            return df.rename(columns={numeric[0]: "sp500_pe_ttm_proxy"})
    return fetch_multpl_series("https://www.multpl.com/s-p-500-pe-ratio", "sp500_pe_ttm_proxy")


def fetch_shiller_cape() -> pd.DataFrame:
    return fetch_multpl_series("https://www.multpl.com/shiller-pe", "shiller_cape")


def fetch_vix() -> pd.DataFrame:
    return fetch_yfinance_close("^VIX", "vix")


def fetch_move() -> pd.DataFrame:
    return fetch_yfinance_close("^MOVE", "move_index")


def fetch_dxy_reference() -> pd.DataFrame:
    try:
        return fetch_dxy_fred_proxy()
    except Exception as exc:  # noqa: PERF203
        report_debug_event(
            "A",
            "fetch_dxy_reference",
            "fred proxy dxy fallback",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )

    try:
        return fetch_dxy_stooq()
    except Exception as exc:  # noqa: PERF203
        report_debug_event(
            "A",
            "fetch_dxy_reference",
            "stooq dxy fallback",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )

    last_error: Exception | None = None
    for ticker in ["DX-Y.NYB", "DX=F"]:
        try:
            history = yf.Ticker(ticker).history(start="2000-01-01", auto_adjust=False)
            if history.empty:
                raise ValueError(f"No price data returned for {ticker}")
            df = history[["Close"]].copy().reset_index()
            date_column = "Date" if "Date" in df.columns else df.columns[0]
            df = df.rename(columns={date_column: "date", "Close": "dxy"})
            normalized = ensure_date_sorted(df[["date", "dxy"]])
            return normalized[normalized["date"] >= pd.Timestamp("2000-01-01")].reset_index(drop=True)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
    cached_csv = OUTPUT_DIR / "05_risk_stress" / "dxy" / "data.csv"
    if cached_csv.exists():
        return rename_single_value_column(fetch_existing_csv(cached_csv), "dxy")
    raise ValueError(f"DXY fetch failed: {last_error}")


def fetch_dxy_fred_proxy() -> pd.DataFrame:
    start_date = pd.Timestamp("2000-01-01")
    anchor = fetch_fred_series_full_history("DTWEXM", "dxy", start_date=start_date)
    last_anchor_date = pd.to_datetime(anchor["date"]).max()

    extension_error: Exception | None = None
    for series_id in ["DTWEXAFEGS", "DTWEXBGS"]:
        try:
            extension = fetch_fred_series_full_history(series_id, "dxy", start_date=start_date)
            overlap = pd.merge(anchor, extension, on="date", how="inner", suffixes=("_anchor", "_extension"))
            overlap = overlap.dropna(subset=["dxy_anchor", "dxy_extension"])
            if overlap.empty:
                raise ValueError(f"No overlap between DTWEXM and {series_id}")

            scale_ratio = (overlap["dxy_anchor"] / overlap["dxy_extension"]).median()
            extension = extension.copy()
            extension["dxy"] = extension["dxy"] * scale_ratio
            stitched = pd.concat(
                [
                    anchor,
                    extension[pd.to_datetime(extension["date"]) > last_anchor_date],
                ],
                ignore_index=True,
            )
            stitched = ensure_date_sorted(stitched, years=None, start_date=start_date)
            if stitched.empty:
                raise ValueError(f"Stitched FRED DXY proxy is empty for {series_id}")
            return stitched
        except Exception as exc:  # noqa: PERF203
            extension_error = exc

    raise ValueError(f"FRED DXY proxy stitch failed: {extension_error}")


def fetch_dxy_stooq() -> pd.DataFrame:
    base_url = "https://stooq.com/q/d/"
    start_date = pd.Timestamp("2000-01-01")
    frames: list[pd.DataFrame] = []
    page = 1
    consecutive_failures = 0

    while True:
        page_url = f"{base_url}?s=usd_i&i=d&l={page}"
        page_response = None
        last_error: Exception | None = None
        for _ in range(3):
            try:
                page_response = requests.get(page_url, timeout=max(TIMEOUT, 30), headers={"User-Agent": "Mozilla/5.0"})
                page_response.raise_for_status()
                break
            except Exception as exc:  # noqa: PERF203
                last_error = exc
        if page_response is None:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                raise ValueError(f"Stooq DXY page fetch failed near page {page}: {last_error}")
            page += 1
            continue

        try:
            tables = pd.read_html(StringIO(page_response.text))
        except Exception as exc:  # noqa: PERF203
            consecutive_failures += 1
            if consecutive_failures >= 2:
                raise ValueError(f"Stooq DXY page parse failed near page {page}: {exc}")
            page += 1
            continue

        history_table = None
        for table in tables:
            columns = [str(column) for column in table.columns]
            if {"No.", "Date", "Close"}.issubset(set(columns)):
                history_table = table
                break
        if history_table is None:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                break
            page += 1
            continue

        consecutive_failures = 0
        frame = history_table[["Date", "Close"]].copy()
        frame.columns = ["date", "dxy"]
        frame["date"] = pd.to_datetime(frame["date"], format="%d %b %Y", errors="coerce")
        frame["dxy"] = pd.to_numeric(frame["dxy"], errors="coerce")
        frame = frame.dropna()
        if not frame.empty:
            frames.append(frame)
            if frame["date"].min() <= start_date:
                break
        page += 1

    if not frames:
        raise ValueError("No DXY history parsed from Stooq")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="first")
    merged = ensure_date_sorted(merged)
    merged = merged[merged["date"] >= start_date].reset_index(drop=True)
    if merged.empty:
        raise ValueError("Stooq DXY history does not cover requested period")
    return merged


def fetch_dxy() -> pd.DataFrame:
    return fetch_dxy_reference()


def fetch_credit_spreads() -> pd.DataFrame:
    return fetch_fred_multi(
        {
            "BAMLC0A0CM": "ig_oas",
            "BAMLH0A0HYM2": "hy_oas",
        }
    )


def fetch_spy_trend() -> pd.DataFrame:
    price = fetch_yfinance_close("SPY", "close")
    price["date"] = pd.to_datetime(price["date"])
    price["ma50"] = price["close"].rolling(50).mean()
    price["ma200"] = price["close"].rolling(200).mean()
    price["price_vs_ma200_pct"] = (price["close"] / price["ma200"] - 1) * 100
    return ensure_date_sorted(price[["date", "close", "ma50", "ma200", "price_vs_ma200_pct"]])


def fetch_sector_rotation() -> pd.DataFrame:
    tickers = {
        "XLK": "technology",
        "XLY": "consumer_discretionary",
        "XLU": "utilities",
        "SPY": "sp500",
    }
    frames = {name: fetch_yfinance_close(ticker, name) for ticker, name in tickers.items()}
    merged = frames["technology"]
    for name in ["consumer_discretionary", "utilities", "sp500"]:
        merged = pd.merge(merged, frames[name], on="date", how="inner")
    merged["xlk_vs_spy"] = merged["technology"] / merged["sp500"] * 100
    merged["xly_vs_spy"] = merged["consumer_discretionary"] / merged["sp500"] * 100
    merged["xlu_vs_spy"] = merged["utilities"] / merged["sp500"] * 100
    return ensure_date_sorted(merged[["date", "xlk_vs_spy", "xly_vs_spy", "xlu_vs_spy"]])


def fetch_volume_price() -> pd.DataFrame:
    df = fetch_yfinance_ohlcv("SPY")
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    return ensure_date_sorted(df[["date", "close", "volume", "volume_ma20"]])


def fetch_commodity_basket() -> pd.DataFrame:
    oil = fetch_commodity_macro_series("oil_wti_daily.csv", "oil", "CL")
    copper = fetch_commodity_macro_series("copper_daily.csv", "copper", "HG")
    soybean = fetch_commodity_macro_series("soybean_daily.csv", "soybean", "S")
    corn = fetch_commodity_macro_series("corn_daily.csv", "corn", "C")
    merged = pd.merge(oil, copper, on="date", how="inner")
    merged = pd.merge(merged, soybean, on="date", how="inner")
    merged = pd.merge(merged, corn, on="date", how="inner")
    return ensure_date_sorted(merged)


def fetch_usd_cnh() -> pd.DataFrame:
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    url = f"https://api.frankfurter.app/{start_date}..{end_date}?from=USD&to=CNY"
    response = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = response.json()
    if "rates" not in payload or not payload["rates"]:
        raise ValueError("No FX data returned from Frankfurter")
    rows = [{"date": date, "usd_cny_proxy": values.get("CNY")} for date, values in payload["rates"].items()]
    return ensure_date_sorted(pd.DataFrame(rows))


def normalize_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = [column for column in result.columns if column != "date"]
    for column in numeric_columns:
        series = result[column].dropna()
        if series.empty:
            continue
        first_value = series.iloc[0]
        if first_value == 0:
            continue
        result[column] = result[column] / first_value * 100
    return result


def plot_dataframe(df: pd.DataFrame, title: str, output_path: Path) -> None:
    numeric_columns = [column for column in df.columns if column != "date"]
    if not numeric_columns:
        return

    rows = len(numeric_columns)
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(4, rows * 2.8)), sharex=True)
    if rows == 1:
        axes = [axes]

    for axis, column in zip(axes, numeric_columns):
        axis.plot(df["date"], df[column], linewidth=1.8, label=column)
        axis.set_ylabel(column)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="upper left", fontsize=9)

    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_core_dashboard(items: list[dict], output_path: Path) -> None:
    if not items:
        return

    cols = 2
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4.2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, item in enumerate(items):
        axis = axes[index]
        df = item["df"]
        numeric_columns = [column for column in df.columns if column != "date"]
        if len(numeric_columns) == 1:
            column = numeric_columns[0]
            axis.plot(df["date"], df[column], linewidth=1.8, color="#1f77b4")
            axis.set_ylabel(column)
        else:
            normalized = normalize_for_dashboard(df)
            for column in numeric_columns:
                axis.plot(normalized["date"], normalized[column], linewidth=1.4, label=column)
            axis.set_ylabel("normalized=100")
            axis.legend(fontsize=7, loc="upper left")
        axis.set_title(item["title"], fontsize=11)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.tick_params(axis="x", rotation=45)

    for index in range(len(items), len(axes)):
        axes[index].axis("off")

    fig.suptitle("US Stock Core Indicator Dashboard", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_guide(guide_path: Path) -> list[dict]:
    lines = guide_path.read_text(encoding="utf-8").splitlines()
    categories: list[dict] = []
    current_category: dict | None = None
    parsing_table = False

    detail_category = None
    detail_indicator = None

    for line in lines:
        stripped = line.strip()

        table_match = re.match(r"^###\s+3\.\d+\s+(.+)$", stripped)
        if table_match:
            current_category = {
                "title": table_match.group(1),
                "slug": category_slug(table_match.group(1)),
                "representative": "",
                "indicators": [],
            }
            categories.append(current_category)
            parsing_table = False
            continue

        if current_category and stripped.startswith("- **代表指标**："):
            current_category["representative"] = stripped.split("：", 1)[1].strip()
            continue

        if current_category and stripped.startswith("| 保留指标 |"):
            parsing_table = True
            continue

        if parsing_table and stripped.startswith("| ---"):
            continue

        if parsing_table and stripped.startswith("| ") and current_category:
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) == 4:
                current_category["indicators"].append(
                    {
                        "name": parts[0],
                        "publicity": parts[1],
                        "importance": parts[2],
                        "reason": parts[3],
                        "details": {},
                    }
                )
            continue

        detail_category_match = re.match(r"^##\s+4\.\d+\s+(.+)$", stripped)
        if detail_category_match:
            detail_category = detail_category_match.group(1)
            detail_indicator = None
            parsing_table = False
            continue

        detail_indicator_match = re.match(r"^###\s+(.+)$", stripped)
        if detail_indicator_match and detail_category:
            detail_indicator = detail_indicator_match.group(1)
            continue

        if detail_category and detail_indicator and stripped.startswith("- **") and "：" in stripped:
            key, value = stripped[2:].split("：", 1)
            key = key.replace("**", "").strip()
            value = value.strip()
            for category_item in categories:
                if category_item["title"] != detail_category:
                    continue
                for indicator in category_item["indicators"]:
                    if indicator["name"] == detail_indicator:
                        indicator["details"][key] = value
                        break
            continue

        if stripped.startswith("## 5. "):
            break

    return categories


def build_registry() -> dict[str, IndicatorConfig]:
    return {
        "联邦基金利率": IndicatorConfig("fed_funds_rate", fetch_fed_funds_rate, "FRED"),
        "美联储资产负债表规模": IndicatorConfig("fed_balance_sheet", lambda: fetch_fred_series("WALCL", "fed_balance_sheet_musd"), "FRED", core=True),
        "美债收益率曲线（2Y-10Y）": IndicatorConfig("yield_curve_2y10y", fetch_yield_curve_fred, "FRED", core=True),
        "SOFR / 隔夜利率": IndicatorConfig("sofr_rate", lambda: fetch_fred_series("SOFR", "sofr"), "FRED"),
        "实际利率（10Y TIPS）": IndicatorConfig("real_rate_10y_tips", fetch_real_rate_tips, "FRED", core=True),
        "M2 货币供应量同比": IndicatorConfig("m2_yoy", fetch_m2_yoy, "FRED"),
        "PCE（个人消费支出平减指数）": IndicatorConfig("pce_inflation", fetch_pce_yoy, "FRED", core=True),
        "CPI（同比 & 核心 CPI）": IndicatorConfig("cpi_inflation", fetch_cpi_yoy, "FRED", core=True),
        "PPI（生产者价格指数）": IndicatorConfig("ppi_inflation", fetch_ppi_yoy, "FRED"),
        "通胀预期（5Y5Y / 密歇根）": IndicatorConfig("inflation_expectations", lambda: fetch_fred_series("T5YIFR", "five_year_five_year_forward"), "FRED"),
        "房租 OER": IndicatorConfig("owner_equivalent_rent", fetch_oer_yoy, "FRED"),
        "大宗商品价格（油、铜、农产品）": IndicatorConfig("commodity_basket", fetch_commodity_basket, "AKShare / commodity_macro cache", note="Uses `output/commodity_macro` cached futures CSVs when available, otherwise fetches via AKShare futures history."),
        "非农就业（NFP）& 失业率": IndicatorConfig("nfp_unemployment", fetch_nfp_unemployment, "FRED", core=True),
        "GDP 增速（实际 & 名义）": IndicatorConfig("gdp_growth", fetch_gdp_growth, "FRED", core=True),
        "ISM 服务业 PMI": IndicatorConfig("ism_services_pmi", fetch_ism_services, "AKShare"),
        "ISM 制造业 PMI": IndicatorConfig("ism_manufacturing_pmi", fetch_ism_manufacturing, "FRED"),
        "零售销售额": IndicatorConfig("retail_sales", fetch_retail_sales, "FRED", core=True),
        "Conference Board LEI": IndicatorConfig("leading_index_proxy", fetch_us_leading_proxy, "AKShare", proxy=True, note="Uses Conference Board consumer confidence as a public cycle proxy in the current environment."),
        "S&P 500 EPS（实际 & 预期）": IndicatorConfig("sp500_eps", fetch_sp500_earnings, "Multpl", proxy=True, note="Fetches historical realized EPS. Forward EPS is not publicly wired yet.", core=True),
        "NTM PE（前向市盈率）": IndicatorConfig("sp500_forward_pe_proxy", fetch_sp500_pe_proxy, "Multpl / existing csv", proxy=True, note="Uses TTM PE public proxy because NTM consensus requires commercial data.", core=True),
        "盈利修正比率（上调 / 下调）": IndicatorConfig("earnings_revision_ratio", None, "Manual", note="Consensus revision feed not wired."),
        "EPS 增速预期": IndicatorConfig("eps_growth_expectation", None, "Manual", note="Forward consensus feed not wired."),
        "净利润率走势": IndicatorConfig("net_margin_trend", None, "Manual", note="Index-level margin feed not wired."),
        "Shiller CAPE": IndicatorConfig("shiller_cape", fetch_shiller_cape, "Multpl", core=True),
        "VIX（波动率指数）": IndicatorConfig("vix", fetch_vix_from_github, "GitHub datasets/finance-vix", core=True),
        "信用利差（IG / HY spread）": IndicatorConfig("credit_spreads", fetch_credit_spreads, "FRED", core=True),
        "MOVE 指数": IndicatorConfig("move_index", fetch_move, "Yahoo Finance"),
        "美元指数（DXY）": IndicatorConfig("dxy", fetch_dxy, "FRED stitched proxy", note="Uses stitched FRED dollar-index proxy history (`DTWEXM` plus newer Fed dollar-index series) to cover 2000-present, then falls back to Stooq, Yahoo Finance, and cached CSV.", core=True),
        "芝加哥联储 NFCI": IndicatorConfig("nfci", lambda: fetch_fred_series("NFCI", "nfci"), "FRED", core=True),
        "OFR 金融压力指数": IndicatorConfig("ofr_financial_stress", None, "Manual", note="Official feed not wired yet."),
        "美股基金资金流（ICI / EPFR）": IndicatorConfig("equity_fund_flows", None, "Manual", note="EPFR/ICI feed not wired."),
        "期货投机净多单（CFTC COT）": IndicatorConfig("cftc_cot", None, "Manual", note="CFTC parsing not wired yet."),
        "Margin Debt（保证金债务）": IndicatorConfig("margin_debt", None, "Manual", note="FINRA monthly parser not wired yet."),
        "股票回购窗口开/关": IndicatorConfig("buyback_window", None, "Manual", note="Needs earnings calendar logic."),
        "外资持仓（TIC 数据）": IndicatorConfig("tic_holdings", None, "Manual", note="TIC parser not wired yet."),
        "散户期权活跃度（0DTE）": IndicatorConfig("odte_activity", None, "Manual", note="Exchange microstructure feed not wired."),
        "200 日 / 50 日均线位置": IndicatorConfig("spy_trend_ma", fetch_sp500_trend_from_existing, "existing csv / AKShare", core=True),
        "涨跌比 & 麦克莱伦振荡器": IndicatorConfig("market_breadth", None, "Manual", note="Breadth universe feed not wired."),
        "52 周新高 vs 新低比率": IndicatorConfig("new_high_new_low_ratio", None, "Manual", note="Breadth universe feed not wired."),
        "高于 200MA 的股票占比": IndicatorConfig("pct_above_200ma", None, "Manual", note="Constituent-level batch feed not wired."),
        "板块轮动（相对强弱 RS）": IndicatorConfig("sector_relative_strength", fetch_sector_rotation_ak, "AKShare", core=True),
        "量价关系（成交量验证）": IndicatorConfig("price_volume_confirmation", fetch_volume_price_ak, "AKShare"),
        "Put/Call 比率": IndicatorConfig("put_call_ratio", None, "Manual", note="CBOE public feed not wired."),
        "AAII 散户多空调查": IndicatorConfig("aaii_sentiment", None, "Manual", note="AAII parser not wired."),
        "NAAIM 主动股票仓位": IndicatorConfig("naaim_exposure", None, "Manual", note="NAAIM parser not wired."),
        "CNN 恐惧贪婪指数": IndicatorConfig("fear_greed_index", None, "Manual", note="CNN public endpoint not wired."),
        "高盛 Bull-Bear 情绪指标": IndicatorConfig("goldman_bull_bear", None, "Manual", note="Commercial source only."),
        "新闻媒体情绪（NLP 指数）": IndicatorConfig("news_sentiment", None, "Manual", note="Commercial/LLM sentiment feed not wired."),
        "全球 PMI 综合（JP Morgan）": IndicatorConfig("global_pmi", None, "Manual", note="Commercial source not wired."),
        "欧元区 / 日本央行政策": IndicatorConfig("ecb_boj_policy", None, "Manual", note="Policy-rate combined public feed not wired."),
        "美债外国持有者结构（TIC）": IndicatorConfig("foreign_treasury_holdings", None, "Manual", note="TIC parser not wired."),
        "人民币汇率（USDCNH）": IndicatorConfig("usd_cnh", fetch_usd_cnh, "Frankfurter", proxy=True, note="Uses USD/CNY public API proxy for USDCNH in the current environment.", core=True),
        "地缘风险事件": IndicatorConfig("geopolitical_risk", None, "Manual", note="Event-driven feed not wired."),
        "商品价格指数（CRB）": IndicatorConfig("crb_index", fetch_crb_recent_series, "StockQ / existing csv", core=True),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report_html(output_root: Path, result_rows: list[dict]) -> str:
    status_counts = pd.Series([row["status"] for row in result_rows]).value_counts().to_dict()
    categories: dict[str, list[dict]] = {}
    for row in result_rows:
        categories.setdefault(row["category"], []).append(row)

    category_sections = []
    for category_name, rows in categories.items():
        rows = sorted(rows, key=lambda item: (not item["representative_of_category"], item["indicator_name"]))
        indicator_cards = []
        for row in rows:
            status_class = f"status-{row['status']}"
            detail_items = []
            for key in ["计算逻辑", "代表含义", "如何影响美股", "参考值"]:
                value = row.get("details", {}).get(key, "")
                if value:
                    detail_items.append(
                        f"<p><strong>{html.escape(key)}：</strong>{html.escape(value)}</p>"
                    )
            img_html = ""
            if row.get("chart_path"):
                rel_chart = Path(row["chart_path"]).relative_to(output_root.parent)
                img_html = f'<a href="{html.escape(rel_chart.as_posix())}" target="_blank"><img src="{html.escape(rel_chart.as_posix())}" alt="{html.escape(row["indicator_name"])}"></a>'

            data_link = ""
            if row.get("data_path"):
                rel_data = Path(row["data_path"]).relative_to(output_root.parent)
                data_link = f'<a href="{html.escape(rel_data.as_posix())}" target="_blank">CSV</a>'

            card = f"""
            <div class="card">
              <div class="card-header">
                <div>
                  <h4>{html.escape(row["indicator_name"])}</h4>
                  <div class="meta-line">
                    <span class="badge {status_class}">{html.escape(row["status"])}</span>
                    <span>重要性: {html.escape(str(row.get("importance", "")))}</span>
                    <span>公开性: {html.escape(str(row.get("publicity", "")))}</span>
                    <span>来源: {html.escape(str(row.get("source", "")))}</span>
                  </div>
                </div>
                <div class="right-note">{'代表指标' if row["representative_of_category"] else ''}</div>
              </div>
              <p class="reason">{html.escape(str(row.get("reason", "")))}</p>
              {img_html}
              <div class="links">{data_link}</div>
              <div class="details">{''.join(detail_items)}</div>
              <p class="note">{html.escape(str(row.get("note", "")))}</p>
              <p class="error">{html.escape(str(row.get("error", "")))}</p>
            </div>
            """
            indicator_cards.append(card)

        category_sections.append(
            f"""
            <section class="category">
              <h2>{html.escape(category_name)}</h2>
              <div class="grid">
                {''.join(indicator_cards)}
              </div>
            </section>
            """
        )

    core_chart_html = ""
    core_chart_path = output_root / "core_dashboard.png"
    if core_chart_path.exists():
        rel_core_chart = core_chart_path.relative_to(output_root.parent)
        core_chart_html = f'<section class="hero"><h2>核心指标汇总</h2><img src="{html.escape(rel_core_chart.as_posix())}" alt="core dashboard"></section>'

    summary_items = "".join(
        f'<span class="summary-chip">{html.escape(key)}: {value}</span>' for key, value in sorted(status_counts.items())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>美股指标抓取汇总</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f7fb; color: #1f2937; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; }}
    h4 {{ margin: 0 0 8px; font-size: 18px; }}
    .top-note {{ color: #4b5563; margin-bottom: 12px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .summary-chip, .badge {{ display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: 12px; background: #e5e7eb; }}
    .hero img, .card img {{ width: 100%; border-radius: 10px; border: 1px solid #d1d5db; background: #fff; }}
    .hero {{ margin-bottom: 28px; }}
    .category {{ margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04); }}
    .card-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .meta-line {{ display: flex; flex-wrap: wrap; gap: 8px; color: #4b5563; font-size: 12px; }}
    .reason, .note, .error {{ font-size: 13px; line-height: 1.6; }}
    .details p {{ margin: 8px 0; font-size: 13px; line-height: 1.6; }}
    .links {{ margin: 10px 0; }}
    .links a {{ color: #2563eb; text-decoration: none; }}
    .right-note {{ font-size: 12px; color: #2563eb; white-space: nowrap; }}
    .status-fetched {{ background: #dcfce7; color: #166534; }}
    .status-fetched_cached {{ background: #dbeafe; color: #1d4ed8; }}
    .status-fetch_failed {{ background: #fee2e2; color: #991b1b; }}
    .status-pending_manual {{ background: #fef3c7; color: #92400e; }}
    .status-skipped {{ background: #e0e7ff; color: #3730a3; }}
    .error {{ color: #b91c1c; }}
    .note {{ color: #6b7280; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>美股指标抓取汇总</h1>
    <p class="top-note">基于手册自动解析类别结构，展示每个指标的抓取状态、说明、图表和数据文件。</p>
    <div class="summary">{summary_items}</div>
    {core_chart_html}
    {''.join(category_sections)}
  </div>
</body>
</html>
"""


def write_html_report(output_root: Path, result_rows: list[dict]) -> Path:
    report_path = output_root / "report.html"
    report_path.write_text(build_report_html(output_root, result_rows), encoding="utf-8")
    return report_path


def save_indicator_result(category_dir: Path, indicator: dict, config: IndicatorConfig, df: pd.DataFrame | None, status: str, error: str = "") -> dict:
    indicator_dir = category_dir / config.slug
    indicator_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "category": indicator["category_title"],
        "category_slug": indicator["category_slug"],
        "indicator_name": indicator["name"],
        "indicator_slug": config.slug,
        "representative_of_category": indicator["name"] == indicator["representative"],
        "importance": indicator.get("importance", ""),
        "publicity": indicator.get("publicity", ""),
        "reason": indicator.get("reason", ""),
        "details": indicator.get("details", {}),
        "source": config.public_source,
        "status": status,
        "proxy": config.proxy,
        "note": config.note,
        "data_path": "",
        "chart_path": "",
        "error": error,
    }

    if df is not None and not df.empty:
        data_path = indicator_dir / "data.csv"
        chart_path = indicator_dir / "chart.png"
        export_df = df.copy()
        export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
        export_df.to_csv(data_path, index=False, encoding="utf-8-sig")
        plot_dataframe(df, indicator["name"], chart_path)
        metadata["data_path"] = str(data_path)
        metadata["chart_path"] = str(chart_path)

    write_json(indicator_dir / "meta.json", metadata)
    return metadata


def load_existing_indicator_data(indicator_dir: Path) -> pd.DataFrame | None:
    data_path = indicator_dir / "data.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    value_columns = [column for column in df.columns if column != "date"]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date"]).dropna(subset=value_columns, how="all").reset_index(drop=True)
    return df if not df.empty else None


def flatten_catalog(categories: list[dict]) -> list[dict]:
    rows = []
    for category_item in categories:
        for indicator in category_item["indicators"]:
            row = indicator.copy()
            row["category_title"] = category_item["title"]
            row["category_slug"] = category_item["slug"]
            row["representative"] = category_item["representative"]
            rows.append(row)
    return rows


def generate_outputs(skip_fetch: bool = False) -> None:
    categories = parse_guide(GUIDE_PATH)
    registry = build_registry()
    output_root = OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    category_roots = {
        item["slug"]: output_root / f"{index:02d}_{item['slug']}"
        for index, item in enumerate(categories, start=1)
    }
    for category_root in category_roots.values():
        category_root.mkdir(parents=True, exist_ok=True)

    catalog_rows = flatten_catalog(categories)
    result_rows = []
    core_dashboard_items = []

    for indicator in catalog_rows:
        config = registry.get(
            indicator["name"],
            IndicatorConfig(stable_slug(indicator["name"]), None, "Manual", note="No fetcher registered."),
        )
        category_dir = category_roots[indicator["category_slug"]]

        if skip_fetch or config.fetcher is None:
            metadata = save_indicator_result(
                category_dir=category_dir,
                indicator=indicator,
                config=config,
                df=None,
                status="pending_manual" if config.fetcher is None else "skipped",
                error="",
            )
            result_rows.append(metadata)
            continue

        try:
            df = config.fetcher()
            metadata = save_indicator_result(
                category_dir=category_dir,
                indicator=indicator,
                config=config,
                df=df,
                status="fetched",
            )
            result_rows.append(metadata)
            if config.core and not df.empty:
                core_dashboard_items.append({"title": indicator["name"], "df": df})
            print(f"[OK] {indicator['category_title']} / {indicator['name']}")
        except Exception as exc:  # noqa: PERF203
            # #region debug-point E:indicator-fail
            report_debug_event(
                "E",
                "generate_outputs",
                "indicator fetch failed",
                {
                    "category": indicator["category_title"],
                    "indicator": indicator["name"],
                    "source": config.public_source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            # #endregion
            existing_df = load_existing_indicator_data(category_dir / config.slug)
            if existing_df is not None:
                fallback_note = f"{exc} | fallback_to_cached_data"
                metadata = save_indicator_result(
                    category_dir=category_dir,
                    indicator=indicator,
                    config=config,
                    df=existing_df,
                    status="fetched_cached",
                    error=fallback_note,
                )
                if config.core and not existing_df.empty:
                    core_dashboard_items.append({"title": indicator["name"], "df": existing_df})
                print(f"[CACHE] {indicator['category_title']} / {indicator['name']}: {exc}")
            else:
                metadata = save_indicator_result(
                    category_dir=category_dir,
                    indicator=indicator,
                    config=config,
                    df=None,
                    status="fetch_failed",
                    error=str(exc),
                )
                print(f"[FAIL] {indicator['category_title']} / {indicator['name']}: {exc}")
            result_rows.append(metadata)

    if core_dashboard_items and not skip_fetch:
        plot_core_dashboard(core_dashboard_items, output_root / "core_dashboard.png")

    catalog_df = pd.DataFrame(result_rows)
    catalog_df.to_csv(output_root / "catalog.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_root / "catalog.json",
        {
            "guide_path": str(GUIDE_PATH),
            "output_root": str(output_root),
            "lookback_years": LOOKBACK_YEARS,
            "items": result_rows,
        },
    )
    write_html_report(output_root, result_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse the guide, fetch supported indicators, and generate charts.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Only parse the guide and create metadata/status files without network fetching.",
    )
    args = parser.parse_args()
    generate_outputs(skip_fetch=args.skip_fetch)


if __name__ == "__main__":
    main()
