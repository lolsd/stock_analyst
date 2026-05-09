from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
STRATEGY_OUTPUT_DIR = BASE_DIR / "strategy" / "output"
OUTPUT_HTML = OUTPUT_DIR / "strategy_dashboard.html"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return normalize_records(df)


def normalize_records(df: pd.DataFrame) -> list[dict]:
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype.kind in {"M", "m"}:
            normalized[column] = normalized[column].astype(str)
    normalized = normalized.where(pd.notna(normalized), None)
    return normalized.to_dict(orient="records")


def relative_to_output(path: Path) -> str:
    return Path(os.path.relpath(path, OUTPUT_DIR)).as_posix()


def load_strategy_payload(strategy_dir: Path) -> dict | None:
    manifest_path = strategy_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)
    summary = read_json(strategy_dir / manifest["artifacts"]["summary"])
    metrics = read_json(strategy_dir / manifest["artifacts"]["metrics"])
    config = read_json(strategy_dir / manifest["artifacts"]["config"])

    nav_rows = read_csv_records(strategy_dir / manifest["artifacts"]["nav"])
    signal_rows = read_csv_records(strategy_dir / manifest["artifacts"]["signals"])
    importance_rows = read_csv_records(strategy_dir / manifest["artifacts"]["features_importance"])
    correlation_rows = read_csv_records(strategy_dir / manifest["artifacts"]["factor_correlation"])

    logs_path = strategy_dir / manifest["artifacts"]["logs"]
    logs_text = logs_path.read_text(encoding="utf-8") if logs_path.exists() else ""

    report_rel = ""
    report_path = strategy_dir / manifest["artifacts"]["report_image"]
    if report_path.exists():
        report_rel = relative_to_output(report_path)

    return {
        "manifest": manifest,
        "summary": summary,
        "metrics": metrics,
        "config": config,
        "nav_rows": nav_rows,
        "signal_rows": signal_rows,
        "importance_rows": importance_rows,
        "correlation_rows": correlation_rows,
        "logs_text": logs_text,
        "report_image": report_rel,
    }


def load_all_strategies() -> list[dict]:
    if not STRATEGY_OUTPUT_DIR.exists():
        return []

    strategies = []
    for child in sorted(STRATEGY_OUTPUT_DIR.iterdir()):
        if not child.is_dir():
            continue
        payload = load_strategy_payload(child)
        if payload is not None:
            strategies.append(payload)
    return strategies


def render_empty_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>量化策略</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f3f6fb; color: #0f172a; }
    .wrap { max-width: 960px; margin: 0 auto; padding: 40px 20px; }
    .card { background: #fff; border: 1px solid #dbe4f0; border-radius: 20px; padding: 24px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06); }
    a { color: #2563eb; text-decoration: none; font-weight: 700; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>量化策略</h1>
      <p>当前还没有可展示的策略结果。请先运行 `strategy` 目录下的策略脚本，生成标准化输出协议文件。</p>
      <p><a href="market_monitor_dashboard.html">返回市场指标监控面板</a></p>
    </div>
  </div>
</body>
</html>
"""


def render_dashboard() -> str:
    strategies = load_all_strategies()
    if not strategies:
        return render_empty_page()

    strategy_map = {
        item["manifest"]["strategy_slug"]: item
        for item in strategies
    }
    default_slug = strategies[0]["manifest"]["strategy_slug"]
    strategy_data_json = json.dumps(strategy_map, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>量化策略</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f3f6fb; color: #0f172a; }}
    a {{ color: inherit; text-decoration: none; }}
    .layout {{ display: flex; min-height: 100vh; }}
    .sidebar {{ width: 240px; background: linear-gradient(180deg, #08152f 0%, #0e2349 100%); color: #dbeafe; padding: 22px 18px; position: sticky; top: 0; height: 100vh; overflow: auto; }}
    .brand {{ display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }}
    .brand-logo {{ width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-weight: 700; background: rgba(255,255,255,0.14); }}
    .brand-title {{ font-size: 16px; font-weight: 700; }}
    .brand-sub {{ font-size: 12px; color: #93c5fd; margin-top: 2px; }}
    .nav {{ display: flex; flex-direction: column; gap: 8px; }}
    .nav a {{ padding: 10px 12px; border-radius: 12px; color: #dbeafe; font-size: 14px; }}
    .nav a:hover {{ background: rgba(255,255,255,0.08); }}
    .main {{ flex: 1; padding: 26px; }}
    .hero, .panel {{ background: #fff; border: 1px solid #dbe4f0; border-radius: 24px; padding: 22px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06); }}
    .hero {{ background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%); }}
    .hero-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }}
    .hero-title {{ margin: 0; font-size: 30px; font-weight: 800; }}
    .hero-desc {{ margin: 8px 0 0; color: #475569; line-height: 1.6; max-width: 760px; }}
    .hero-actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .hero-link {{ display: inline-flex; align-items: center; justify-content: center; min-height: 42px; padding: 10px 14px; border-radius: 12px; background: #eff6ff; color: #1d4ed8; font-size: 13px; font-weight: 700; border: 1px solid #bfdbfe; }}
    .hero-tags {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .hero-tags span {{ font-size: 12px; border-radius: 999px; background: #e2e8f0; color: #334155; padding: 6px 10px; }}
    .section {{ margin-top: 22px; }}
    .section-title {{ margin: 0 0 14px; font-size: 22px; font-weight: 800; }}
    .toolbar {{ display: grid; grid-template-columns: 1.3fr 1fr 1fr auto auto; gap: 12px; align-items: end; }}
    .field label {{ display: block; margin-bottom: 6px; font-size: 12px; color: #64748b; font-weight: 700; }}
    .field select, .field input {{ width: 100%; min-height: 42px; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 12px; background: #fff; color: #0f172a; font-size: 14px; }}
    .btn {{ min-height: 42px; padding: 10px 14px; border-radius: 12px; border: 1px solid #cbd5e1; background: #fff; font-size: 13px; font-weight: 700; cursor: pointer; }}
    .btn.primary {{ border-color: #2563eb; background: #2563eb; color: #fff; }}
    .grid-cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .metric-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; }}
    .metric-label {{ color: #64748b; font-size: 12px; }}
    .metric-value {{ margin-top: 10px; font-size: 28px; font-weight: 800; }}
    .metric-sub {{ margin-top: 8px; color: #475569; font-size: 12px; }}
    .two-col {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; }}
    .chart-panel {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 22px; padding: 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
    .chart-title {{ margin: 0 0 8px; font-size: 18px; font-weight: 800; }}
    .chart-desc {{ margin: 0 0 14px; color: #64748b; font-size: 13px; line-height: 1.6; }}
    .chart-box {{ width: 100%; height: 360px; border: 1px solid #dbe4f0; border-radius: 14px; background: #fff; }}
    .chart-box.tall {{ height: 420px; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; }}
    .metrics-table th, .metrics-table td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 13px; }}
    .metrics-table th {{ color: #64748b; font-weight: 700; background: #f8fafc; }}
    .config-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .config-group {{ border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; background: #fbfdff; }}
    .config-group h3 {{ margin: 0 0 12px; font-size: 16px; }}
    .config-item {{ display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px dashed #e2e8f0; }}
    .config-item:last-child {{ border-bottom: 0; }}
    .config-item label {{ color: #64748b; font-size: 13px; }}
    .config-item span {{ font-size: 13px; font-weight: 700; color: #0f172a; text-align: right; }}
    .mono-block {{ white-space: pre-wrap; background: #0f172a; color: #dbeafe; border-radius: 18px; padding: 16px; font-size: 12px; line-height: 1.6; overflow: auto; max-height: 360px; }}
    .report-img {{ width: 100%; border-radius: 16px; border: 1px solid #dbe4f0; background: #fff; }}
    .empty {{ padding: 24px; border: 1px dashed #cbd5e1; border-radius: 16px; color: #64748b; background: #f8fafc; }}
    .warn-list {{ margin: 0; padding-left: 18px; color: #92400e; }}
    .warn-list li {{ margin: 6px 0; }}
    @media (max-width: 1280px) {{
      .grid-cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .config-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 960px) {{
      .layout {{ display: block; }}
      .sidebar {{ width: auto; height: auto; position: sticky; z-index: 20; border-bottom: 1px solid rgba(255,255,255,0.08); }}
      .nav {{ flex-direction: row; overflow-x: auto; }}
      .main {{ padding: 18px; }}
      .hero-top {{ flex-direction: column; }}
      .two-col, .config-grid, .grid-cards, .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-logo">QS</div>
        <div>
          <div class="brand-title">Quant Strategy</div>
          <div class="brand-sub">量化策略与回测结果</div>
        </div>
      </div>
      <nav class="nav">
        <a href="market_monitor_dashboard.html">市场监控</a>
        <a href="#overview">概览</a>
        <a href="#charts">图表</a>
        <a href="#metrics">指标</a>
        <a href="#config">配置</a>
        <a href="#logs">日志</a>
      </nav>
    </aside>
    <main class="main">
      <section class="hero" id="overview">
        <div class="hero-top">
          <div>
            <h1 class="hero-title">量化策略</h1>
            <p class="hero-desc">当前版本先接入遵守标准输出协议的策略结果，支持策略切换、日期筛选、净值/回撤、概率/仓位、因子重要性、相关性、配置面板与日志查看。</p>
            <div class="hero-tags">
              <span>标准输出协议</span>
              <span>静态生成页面</span>
              <span>前端时间筛选</span>
            </div>
          </div>
          <div class="hero-actions">
            <a class="hero-link" href="market_monitor_dashboard.html">返回市场监控</a>
          </div>
        </div>
      </section>

      <section class="section panel">
        <div class="toolbar">
          <div class="field">
            <label for="strategy-select">策略</label>
            <select id="strategy-select"></select>
          </div>
          <div class="field">
            <label for="date-start">开始日期</label>
            <input id="date-start" type="date">
          </div>
          <div class="field">
            <label for="date-end">结束日期</label>
            <input id="date-end" type="date">
          </div>
          <button class="btn primary" id="apply-range" type="button">应用区间</button>
          <button class="btn" id="reset-range" type="button">重置</button>
        </div>
      </section>

      <section class="section" id="summary">
        <h2 class="section-title">策略概览</h2>
        <div class="grid-cards" id="summary-cards"></div>
      </section>

      <section class="section" id="charts">
        <div class="two-col">
          <div class="chart-panel">
            <h3 class="chart-title">净值与回撤</h3>
            <p class="chart-desc">展示策略净值、基准净值以及双边回撤，支持内部缩放和下方日期筛选。</p>
            <div class="chart-box tall" id="nav-chart"></div>
          </div>
          <div class="chart-panel">
            <h3 class="chart-title">概率与仓位</h3>
            <p class="chart-desc">展示三分类概率滚动走势以及策略仓位，用于观察信号强弱变化。</p>
            <div class="chart-box tall" id="signal-chart"></div>
          </div>
        </div>
        <div class="two-col section">
          <div class="chart-panel">
            <h3 class="chart-title">因子重要性</h3>
            <p class="chart-desc">展示外部重要性与模型内置重要性，当前 `func1` 使用标准协议中的 `features_importance.csv`。</p>
            <div class="chart-box" id="importance-chart"></div>
          </div>
          <div class="chart-panel">
            <h3 class="chart-title">因子相关性</h3>
            <p class="chart-desc">当前展示 Top 因子相关性热力图，数据来自 `factor_correlation.csv` 长表。</p>
            <div class="chart-box" id="correlation-chart"></div>
          </div>
        </div>
      </section>

      <section class="section" id="metrics">
        <div class="two-col">
          <div class="chart-panel">
            <h3 class="chart-title">绩效指标</h3>
            <p class="chart-desc">直接读取协议中的 `metrics.json`，统一展示策略和基准。</p>
            <div id="metrics-table-wrap"></div>
          </div>
          <div class="chart-panel">
            <h3 class="chart-title">运行告警</h3>
            <p class="chart-desc">这里汇总 `manifest.json` 中的 `warnings` 和运行状态。</p>
            <div id="warning-wrap"></div>
          </div>
        </div>
      </section>

      <section class="section" id="config">
        <div class="chart-panel">
          <h3 class="chart-title">配置面板</h3>
          <p class="chart-desc">当前先展示标准协议 `config.json` 中的参数；下一阶段可接入实际重跑。</p>
          <div class="config-grid" id="config-grid"></div>
        </div>
      </section>

      <section class="section">
        <div class="two-col">
          <div class="chart-panel">
            <h3 class="chart-title">原始报告图</h3>
            <p class="chart-desc">保留策略脚本生成的原始报告图，作为结构化图表之外的兜底展示。</p>
            <div id="report-wrap"></div>
          </div>
          <div class="chart-panel" id="logs">
            <h3 class="chart-title">运行日志</h3>
            <p class="chart-desc">直接读取标准协议中的 `logs.txt`，便于后续接入运行详情抽屉。</p>
            <div id="logs-wrap"></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <script>
    const STRATEGY_DATA = {strategy_data_json};
    const DEFAULT_SLUG = {json.dumps(default_slug, ensure_ascii=False)};

    const strategySelect = document.getElementById("strategy-select");
    const dateStartInput = document.getElementById("date-start");
    const dateEndInput = document.getElementById("date-end");

    const navChart = echarts.init(document.getElementById("nav-chart"));
    const signalChart = echarts.init(document.getElementById("signal-chart"));
    const importanceChart = echarts.init(document.getElementById("importance-chart"));
    const correlationChart = echarts.init(document.getElementById("correlation-chart"));

    function formatPct(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return `${{(Number(value) * 100).toFixed(2)}}%`;
    }}

    function formatNum(value, digits = 3) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(digits);
    }}

    function formatSignedPct(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      const num = Number(value);
      const sign = num > 0 ? "+" : "";
      return `${{sign}}${{(num * 100).toFixed(2)}}%`;
    }}

    function formatDate(value) {{
      if (!value) return "--";
      return value;
    }}

    function buildSummaryCards(payload) {{
      const summary = payload.summary;
      const benchmark = summary.benchmark || {{}};
      const classification = summary.classification || {{}};
      const generatedAt = payload.manifest.generated_at || "--";
      return [
        {{
          label: "策略总收益",
          value: formatPct(summary.headline.total_return),
          sub: `年化 ${{formatPct(summary.headline.annual_return)}} / 夏普 ${{formatNum(summary.headline.sharpe)}}`,
        }},
        {{
          label: "最大回撤",
          value: formatPct(summary.headline.max_drawdown),
          sub: `Calmar ${{formatNum(summary.headline.calmar)}} / 胜率 ${{formatPct(summary.headline.win_rate)}}`,
        }},
        {{
          label: "基准表现",
          value: formatPct(benchmark.total_return),
          sub: `${{benchmark.name || "基准"}} 年化 ${{formatPct(benchmark.annual_return)}}`,
        }},
        {{
          label: "分类效果",
          value: formatNum(classification.accuracy, 3),
          sub: `Macro F1 ${{formatNum(classification.macro_f1, 3)}} / 更新于 ${{generatedAt}}`,
        }},
      ];
    }}

    function renderSummaryCards(payload) {{
      const cards = buildSummaryCards(payload);
      document.getElementById("summary-cards").innerHTML = cards.map((item) => `
        <div class="metric-card">
          <div class="metric-label">${{item.label}}</div>
          <div class="metric-value">${{item.value}}</div>
          <div class="metric-sub">${{item.sub}}</div>
        </div>
      `).join("");
    }}

    function renderMetricsTable(payload) {{
      const metrics = payload.metrics || {{}};
      const rows = metrics.rows || [];
      if (!rows.length) {{
        document.getElementById("metrics-table-wrap").innerHTML = '<div class="empty">暂无指标数据</div>';
        return;
      }}
      document.getElementById("metrics-table-wrap").innerHTML = `
        <table class="metrics-table">
          <thead>
            <tr>
              <th>名称</th>
              <th>年化收益</th>
              <th>波动</th>
              <th>夏普</th>
              <th>最大回撤</th>
              <th>Calmar</th>
              <th>胜率</th>
              <th>总收益</th>
            </tr>
          </thead>
          <tbody>
            ${{
              rows.map((row) => `
                <tr>
                  <td>${{row.name}}</td>
                  <td>${{formatPct(row.annual_return)}}</td>
                  <td>${{formatPct(row.annual_volatility)}}</td>
                  <td>${{formatNum(row.sharpe)}}</td>
                  <td>${{formatPct(row.max_drawdown)}}</td>
                  <td>${{formatNum(row.calmar)}}</td>
                  <td>${{formatPct(row.win_rate)}}</td>
                  <td>${{formatPct(row.total_return)}}</td>
                </tr>
              `).join("")
            }}
          </tbody>
        </table>
      `;
    }}

    function renderWarnings(payload) {{
      const warnings = payload.manifest.warnings || [];
      const status = payload.manifest.status || "--";
      if (!warnings.length) {{
        document.getElementById("warning-wrap").innerHTML = `<div class="empty">状态：${{status}}，当前没有额外告警。</div>`;
        return;
      }}
      document.getElementById("warning-wrap").innerHTML = `
        <div class="empty">
          <div style="margin-bottom:10px;font-weight:700;">状态：${{status}}</div>
          <ul class="warn-list">${{warnings.map((item) => `<li>${{item}}</li>`).join("")}}</ul>
        </div>
      `;
    }}

    function renderConfig(payload) {{
      const groups = (payload.config && payload.config.groups) || [];
      const root = document.getElementById("config-grid");
      if (!groups.length) {{
        root.innerHTML = '<div class="empty">暂无配置面板</div>';
        return;
      }}
      root.innerHTML = groups.map((group) => `
        <div class="config-group">
          <h3>${{group.group_name}}</h3>
          ${{
            (group.items || []).map((item) => `
              <div class="config-item">
                <label>${{item.label}}</label>
                <span>${{item.value}}</span>
              </div>
            `).join("")
          }}
        </div>
      `).join("");
    }}

    function renderReport(payload) {{
      const root = document.getElementById("report-wrap");
      if (!payload.report_image) {{
        root.innerHTML = '<div class="empty">暂无报告图片</div>';
        return;
      }}
      root.innerHTML = `<img class="report-img" src="${{payload.report_image}}" alt="strategy report">`;
    }}

    function renderLogs(payload) {{
      const root = document.getElementById("logs-wrap");
      root.innerHTML = `<div class="mono-block">${{payload.logs_text || "暂无日志"}}</div>`;
    }}

    function getDateBounds(navRows) {{
      if (!navRows.length) return {{ start: "", end: "" }};
      return {{ start: navRows[0].date, end: navRows[navRows.length - 1].date }};
    }}

    function filterRows(rows, start, end) {{
      return (rows || []).filter((row) => {{
        const value = row.date;
        if (!value) return false;
        if (start && value < start) return false;
        if (end && value > end) return false;
        return true;
      }});
    }}

    function renderNavChart(navRows) {{
      if (!navRows.length) {{
        navChart.setOption({{ title: {{ text: "暂无净值数据" }} }}, true);
        return;
      }}
      const buyPoints = [];
      const sellPoints = [];
      const positionDeltaMap = new Map();
      navRows.forEach((row, index) => {{
        const currentPosition = Number(row.position || 0);
        const previousPosition = index > 0 ? Number(navRows[index - 1].position || 0) : 0;
        const delta = currentPosition - previousPosition;
        positionDeltaMap.set(row.date, delta);
        if (delta > 1e-9) {{
          buyPoints.push({{
            name: "买入",
            // ECharts scatter uses `value: [x, y]` as the coordinate.
            value: [row.date, Number(row.benchmark_nav)],
            positionValue: currentPosition,
            positionDelta: delta,
            benchmarkNav: Number(row.benchmark_nav),
          }});
        }} else if (delta < -1e-9) {{
          sellPoints.push({{
            name: "卖出",
            value: [row.date, Number(row.benchmark_nav)],
            positionValue: currentPosition,
            positionDelta: delta,
            benchmarkNav: Number(row.benchmark_nav),
          }});
        }}
      }});

      navChart.setOption({{
        tooltip: {{
          trigger: "axis",
          formatter: (params) => {{
            const axisParams = Array.isArray(params) ? params : [params];
            const axisValue = axisParams[0] ? axisParams[0].axisValue : "";
            const row = navRows.find((item) => item.date === axisValue) || {{}};
            const position = Number(row.position || 0);
            const delta = positionDeltaMap.get(axisValue) || 0;
            const actionText = delta > 1e-9 ? `买入 ${{formatSignedPct(delta)}}` : delta < -1e-9 ? `卖出 ${{formatSignedPct(Math.abs(delta) * -1)}}` : "无调仓";
            const lines = [
              `<strong>${{axisValue}}</strong>`,
              `当前仓位：${{formatPct(position)}}`,
              `操作仓位：${{actionText}}`,
            ];
            axisParams.forEach((item) => {{
              if (item.seriesName === "买入点" || item.seriesName === "卖出点") return;
              const isDrawdown = item.seriesName.includes("回撤");
              const valueText = isDrawdown ? formatPct(item.value) : formatNum(item.value, 3);
              lines.push(`${{item.marker}}${{item.seriesName}}：${{valueText}}`);
            }});
            return lines.join("<br>");
          }}
        }},
        legend: {{ top: 0 }},
        grid: {{ left: 54, right: 40, top: 40, bottom: 64 }},
        dataZoom: [{{ type: "inside" }}, {{ type: "slider", height: 24, bottom: 16 }}],
        xAxis: {{ type: "category", boundaryGap: false, data: navRows.map((row) => row.date) }},
        yAxis: [
          {{ type: "value", scale: true, axisLabel: {{ formatter: (value) => value.toFixed(2) + "x" }} }},
          {{ type: "value", scale: true, axisLabel: {{ formatter: (value) => (value * 100).toFixed(0) + "%" }} }}
        ],
        series: [
          {{ name: "策略净值", type: "line", smooth: true, showSymbol: false, data: navRows.map((row) => row.strategy_nav), lineStyle: {{ width: 2, color: "#2563eb" }} }},
          {{ name: "纳斯达克指数", type: "line", smooth: true, showSymbol: false, data: navRows.map((row) => row.benchmark_nav), lineStyle: {{ width: 2, color: "#16a34a" }} }},
          {{ name: "策略回撤", type: "line", smooth: true, showSymbol: false, yAxisIndex: 1, data: navRows.map((row) => row.strategy_drawdown), lineStyle: {{ width: 1.5, color: "#dc2626", type: "dashed" }} }},
          {{ name: "纳指回撤", type: "line", smooth: true, showSymbol: false, yAxisIndex: 1, data: navRows.map((row) => row.benchmark_drawdown), lineStyle: {{ width: 1.5, color: "#94a3b8", type: "dashed" }} }},
          {{
            name: "买入点",
            type: "scatter",
            symbol: "circle",
            symbolSize: 11,
            data: buyPoints,
            itemStyle: {{ color: "#dc2626" }},
            tooltip: {{
              formatter: (item) => `买入<br>${{item.data.value[0]}}<br>纳指：${{formatNum(item.data.value[1], 3)}}<br>仓位：${{formatPct(item.data.positionValue)}}<br>加仓：${{formatPct(item.data.positionDelta)}}`
            }}
          }},
          {{
            name: "卖出点",
            type: "scatter",
            symbol: "diamond",
            symbolSize: 11,
            data: sellPoints,
            itemStyle: {{ color: "#16a34a" }},
            tooltip: {{
              formatter: (item) => `卖出<br>${{item.data.value[0]}}<br>纳指：${{formatNum(item.data.value[1], 3)}}<br>仓位：${{formatPct(item.data.positionValue)}}<br>减仓：${{formatPct(Math.abs(item.data.positionDelta))}}`
            }}
          }}
        ]
      }}, true);
    }}

    function renderSignalChart(signalRows) {{
      if (!signalRows.length) {{
        signalChart.setOption({{ title: {{ text: "暂无信号数据" }} }}, true);
        return;
      }}
      signalChart.setOption({{
        tooltip: {{ trigger: "axis" }},
        legend: {{ top: 0 }},
        grid: {{ left: 54, right: 40, top: 40, bottom: 64 }},
        dataZoom: [{{ type: "inside" }}, {{ type: "slider", height: 24, bottom: 16 }}],
        xAxis: {{ type: "category", boundaryGap: false, data: signalRows.map((row) => row.date) }},
        yAxis: [
          {{ type: "value", min: 0, max: 1 }},
          {{ type: "value", min: 0, max: 1.2 }}
        ],
        series: [
          {{ name: "P(Up)", type: "line", showSymbol: false, smooth: true, data: signalRows.map((row) => row.p_up), lineStyle: {{ color: "#16a34a", width: 1.8 }} }},
          {{ name: "P(Flat)", type: "line", showSymbol: false, smooth: true, data: signalRows.map((row) => row.p_flat), lineStyle: {{ color: "#f59e0b", width: 1.6 }} }},
          {{ name: "P(Down)", type: "line", showSymbol: false, smooth: true, data: signalRows.map((row) => row.p_down), lineStyle: {{ color: "#dc2626", width: 1.6 }} }},
          {{ name: "仓位", type: "line", showSymbol: false, smooth: true, yAxisIndex: 1, data: signalRows.map((row) => row.position), lineStyle: {{ color: "#2563eb", width: 2 }} }}
        ]
      }}, true);
    }}

    function renderImportanceChart(rows) {{
      const subset = (rows || []).slice(0, 15).reverse();
      if (!subset.length) {{
        importanceChart.setOption({{ title: {{ text: "暂无重要性数据" }} }}, true);
        return;
      }}
      importanceChart.setOption({{
        tooltip: {{ trigger: "axis", axisPointer: {{ type: "shadow" }} }},
        grid: {{ left: 120, right: 24, top: 16, bottom: 28 }},
        xAxis: {{ type: "value" }},
        yAxis: {{ type: "category", data: subset.map((row) => row.feature) }},
        series: [
          {{
            name: "重要性",
            type: "bar",
            data: subset.map((row) => row.importance_score),
            itemStyle: {{ color: "#2563eb" }},
          }}
        ]
      }}, true);
    }}

    function renderCorrelationChart(rows) {{
      if (!rows.length) {{
        correlationChart.setOption({{ title: {{ text: "暂无相关性数据" }} }}, true);
        return;
      }}
      const xLabels = Array.from(new Set(rows.map((row) => row.feature_x)));
      const yLabels = Array.from(new Set(rows.map((row) => row.feature_y)));
      const values = rows.map((row) => [
        xLabels.indexOf(row.feature_x),
        yLabels.indexOf(row.feature_y),
        Number(row.corr)
      ]);
      correlationChart.setOption({{
        tooltip: {{
          position: "top",
          formatter: (params) => `${{xLabels[params.value[0]]}} / ${{yLabels[params.value[1]]}}<br>相关系数: ${{Number(params.value[2]).toFixed(3)}}`
        }},
        grid: {{ left: 110, right: 24, top: 20, bottom: 80 }},
        xAxis: {{ type: "category", data: xLabels, axisLabel: {{ rotate: 35 }} }},
        yAxis: {{ type: "category", data: yLabels }},
        visualMap: {{
          min: -1,
          max: 1,
          calculable: true,
          orient: "horizontal",
          left: "center",
          bottom: 18,
          inRange: {{ color: ["#f85149", "#f8fafc", "#16a34a"] }}
        }},
        series: [
          {{
            type: "heatmap",
            data: values,
            label: {{ show: true, formatter: (params) => Number(params.value[2]).toFixed(2), color: "#0f172a", fontSize: 10 }},
            emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: "rgba(0,0,0,0.18)" }} }}
          }}
        ]
      }}, true);
    }}

    function applyRangeToStrategy(payload) {{
      const navRows = filterRows(payload.nav_rows, dateStartInput.value, dateEndInput.value);
      const signalRows = filterRows(payload.signal_rows, dateStartInput.value, dateEndInput.value);
      renderNavChart(navRows);
      renderSignalChart(signalRows);
      renderImportanceChart(payload.importance_rows || []);
      renderCorrelationChart(payload.correlation_rows || []);
    }}

    function renderStrategy(slug) {{
      const payload = STRATEGY_DATA[slug];
      if (!payload) return;

      renderSummaryCards(payload);
      renderMetricsTable(payload);
      renderWarnings(payload);
      renderConfig(payload);
      renderReport(payload);
      renderLogs(payload);

      const bounds = getDateBounds(payload.nav_rows || []);
      dateStartInput.min = bounds.start;
      dateStartInput.max = bounds.end;
      dateEndInput.min = bounds.start;
      dateEndInput.max = bounds.end;

      if (!dateStartInput.value || dateStartInput.value < bounds.start || dateStartInput.value > bounds.end) {{
        dateStartInput.value = payload.manifest.date_range.test_start || bounds.start;
      }}
      if (!dateEndInput.value || dateEndInput.value < bounds.start || dateEndInput.value > bounds.end) {{
        dateEndInput.value = payload.manifest.date_range.test_end || bounds.end;
      }}

      applyRangeToStrategy(payload);
    }}

    function initSelector() {{
      const entries = Object.entries(STRATEGY_DATA);
      strategySelect.innerHTML = entries.map(([slug, payload]) => `<option value="${{slug}}">${{payload.manifest.strategy_name}} (${{slug}})</option>`).join("");
      strategySelect.value = DEFAULT_SLUG;
      strategySelect.addEventListener("change", () => renderStrategy(strategySelect.value));
      document.getElementById("apply-range").addEventListener("click", () => renderStrategy(strategySelect.value));
      document.getElementById("reset-range").addEventListener("click", () => {{
        const payload = STRATEGY_DATA[strategySelect.value];
        const bounds = getDateBounds(payload.nav_rows || []);
        dateStartInput.value = payload.manifest.date_range.test_start || bounds.start;
        dateEndInput.value = payload.manifest.date_range.test_end || bounds.end;
        renderStrategy(strategySelect.value);
      }});
      renderStrategy(DEFAULT_SLUG);
    }}

    initSelector();
    window.addEventListener("resize", () => {{
      navChart.resize();
      signalChart.resize();
      importanceChart.resize();
      correlationChart.resize();
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(render_dashboard(), encoding="utf-8")
    print(f"Generated: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
