"""
XGBoost 多因子量化策略
- 训练集：2015-01-01 ~ 2020-12-31
- 测试集：2021-01-01 ~ 2024-12-31
- 因子：VIX、PE分位点、纳斯达克、美元指数 + 衍生因子
- 输出：净值曲线、回撤、绩效指标、因子重要性、相关性矩阵
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import json
from datetime import datetime, timezone
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings('ignore')

XGBOOST_IMPORT_ERROR = None
try:
    from xgboost import XGBClassifier
except Exception as exc:  # noqa: PERF203
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

SHAP_IMPORT_ERROR = None
try:
    import shap
except Exception as exc:  # noqa: PERF203
    shap = None
    SHAP_IMPORT_ERROR = exc

np.random.seed(42)

SCHEMA_VERSION = "1.0"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STRATEGY_SLUG = Path(__file__).stem
STRATEGY_NAME = ((__doc__ or "").strip().splitlines() or [STRATEGY_SLUG])[0].strip() or STRATEGY_SLUG

OUTPUT_ROOT = BASE_DIR / "output" / STRATEGY_SLUG
ARTIFACTS_DIR = OUTPUT_ROOT / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "report.png"
LOG_PATH = OUTPUT_ROOT / "logs.txt"

TRAIN_START = "2015-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2024-12-31"

MODEL_LABEL = "XGBoost" if XGBClassifier is not None else "RandomForest Fallback"
IMPORTANCE_LABEL = "SHAP" if shap is not None else "Permutation"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
GENERATED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
LOG_LINES: list[str] = []


def log(message: str = "") -> None:
    print(message)
    LOG_LINES.append(message)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

# ═══════════════════════════════════════════════════
# 1. 读取真实数据
# ═══════════════════════════════════════════════════
def read_series(
    source: Union[str, Path],
    *,
    value_column: str,
    output_name: str,
    date_column: str = "date",
) -> pd.DataFrame:
    df = pd.read_csv(source)
    lowered = {column.lower(): column for column in df.columns}
    resolved_date_col = lowered.get(date_column.lower())
    resolved_value_col = lowered.get(value_column.lower())
    if resolved_date_col is None or resolved_value_col is None:
        raise ValueError(f"无法在 {source} 中找到列 {date_column}/{value_column}")
    result = df[[resolved_date_col, resolved_value_col]].copy()
    result.columns = ["date", output_name]
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result[output_name] = pd.to_numeric(result[output_name], errors="coerce")
    return result.dropna().sort_values("date").drop_duplicates(subset=["date"], keep="last")


def load_vix_series() -> pd.DataFrame:
    local_path = PROJECT_DIR / "output" / "us_stock_indicators" / "05_risk_stress" / "vix" / "data.csv"
    if local_path.exists():
        local_df = read_series(local_path, value_column="vix", output_name="VIX")
        if not local_df.empty and local_df["date"].min() <= pd.Timestamp(TRAIN_START):
            return local_df

    # 本地 VIX 缓存当前只有近几年，这里回退到项目原有数据源补齐训练期历史。
    github_url = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
    return read_series(github_url, value_column="close", output_name="VIX", date_column="date")


def load_real_market_data() -> pd.DataFrame:
    output_root = PROJECT_DIR / "output"
    nasdaq = read_series(output_root / "nasdaq_daily.csv", value_column="value", output_name="NASDAQ")
    pe_pct = read_series(output_root / "nasdaq_pe_percentile_daily.csv", value_column="value", output_name="PE_PCT")
    dxy = read_series(
        output_root / "us_stock_indicators" / "05_risk_stress" / "dxy" / "data.csv",
        value_column="dxy",
        output_name="DXY",
    )
    vix = load_vix_series()

    raw = nasdaq.set_index("date").sort_index()
    raw = raw.join(vix.set_index("date"), how="left")
    raw = raw.join(pe_pct.set_index("date"), how="left")
    raw = raw.join(dxy.set_index("date"), how="left")
    raw = raw.sort_index().ffill()
    raw = raw.loc[:TEST_END].dropna()

    if raw["PE_PCT"].max() > 1.5:
        raw["PE_PCT"] = raw["PE_PCT"] / 100.0

    return raw


raw = load_real_market_data()

# ═══════════════════════════════════════════════════
# 2. 因子工程（丰富特征）
# ═══════════════════════════════════════════════════
def build_features(df):
    f = pd.DataFrame(index=df.index)

    # ── VIX 因子族
    f['vix_level']     = df['VIX']
    f['vix_zscore']    = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
    f['vix_mom5']      = df['VIX'].pct_change(5)
    f['vix_mom20']     = df['VIX'].pct_change(20)
    f['vix_term']      = df['VIX'].rolling(5).mean() / df['VIX'].rolling(60).mean() - 1  # 期限结构代理

    # ── PE 因子族
    f['pe_pct']        = df['PE_PCT']
    f['pe_pct_chg20']  = df['PE_PCT'].diff(20)
    f['pe_zscore']     = (df['PE_PCT'] - df['PE_PCT'].rolling(252).mean()) / df['PE_PCT'].rolling(252).std()

    # ── 纳斯达克动量/技术因子
    ret1  = df['NASDAQ'].pct_change(1)
    ret5  = df['NASDAQ'].pct_change(5)
    ret20 = df['NASDAQ'].pct_change(20)
    ret60 = df['NASDAQ'].pct_change(60)
    ma20  = df['NASDAQ'].rolling(20).mean()
    ma60  = df['NASDAQ'].rolling(60).mean()
    ma120 = df['NASDAQ'].rolling(120).mean()
    std20 = df['NASDAQ'].rolling(20).std()

    f['nasdaq_ret1']   = ret1
    f['nasdaq_ret5']   = ret5
    f['nasdaq_ret20']  = ret20
    f['nasdaq_ret60']  = ret60
    f['nasdaq_ma_ratio']  = ma20 / ma60 - 1          # 均线偏离
    f['nasdaq_ma_ratio2'] = ma60 / ma120 - 1
    f['nasdaq_rsi']    = _rsi(df['NASDAQ'], 14)
    f['nasdaq_vol20']  = std20 / ma20                 # 波动率/价格
    f['nasdaq_vol_ratio'] = df['NASDAQ'].rolling(5).std() / df['NASDAQ'].rolling(60).std()

    # ── 美元指数因子族
    f['dxy_ret5']      = df['DXY'].pct_change(5)
    f['dxy_ret20']     = df['DXY'].pct_change(20)
    f['dxy_zscore']    = (df['DXY'] - df['DXY'].rolling(252).mean()) / df['DXY'].rolling(252).std()
    f['dxy_trend']     = df['DXY'].rolling(10).mean() / df['DXY'].rolling(60).mean() - 1

    # ── 交叉因子（组合信号）
    f['vix_x_dxy']     = f['vix_zscore'] * f['dxy_zscore']         # 双重避险
    f['pe_x_nasdaq']   = f['pe_pct'] * f['nasdaq_ma_ratio']         # 估值×趋势
    f['fear_index']    = f['vix_zscore'] + f['dxy_zscore'] - f['nasdaq_ma_ratio']

    return f

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

feat = build_features(raw)
FACTOR_COLS = feat.columns.tolist()

# ═══════════════════════════════════════════════════
# 3. 标签：预测未来5日涨跌（三分类：涨/平/跌）
# ═══════════════════════════════════════════════════
HORIZON = 5
fwd_ret = raw['NASDAQ'].pct_change(HORIZON).shift(-HORIZON)

THRESH = 0.008   # 阈值：±0.8% 区分涨跌
label = pd.cut(fwd_ret, bins=[-np.inf, -THRESH, THRESH, np.inf], labels=[0, 1, 2])
label = label.astype(float)

data = feat.copy()
data['label'] = label
data['fwd_ret'] = fwd_ret
data['nasdaq_ret_raw'] = raw['NASDAQ'].pct_change()
data = data.dropna()
data = data[(data.index >= TRAIN_START) & (data.index <= TEST_END)]

# ═══════════════════════════════════════════════════
# 4. 训练/测试分割
# ═══════════════════════════════════════════════════
train = data[data.index <= TRAIN_END]
test  = data[data.index >= TEST_START]

X_train = train[FACTOR_COLS]
y_train = train['label'].astype(int)
X_test  = test[FACTOR_COLS]
y_test  = test['label'].astype(int)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

log(f"训练集：{train.index[0].date()} ~ {train.index[-1].date()}  ({len(train)} days)")
log(f"测试集：{test.index[0].date()}  ~ {test.index[-1].date()}  ({len(test)} days)")
log(f"模型：{MODEL_LABEL}")
if XGBOOST_IMPORT_ERROR is not None:
    log(f"XGBoost 不可用，已自动回退到 RandomForest：{XGBOOST_IMPORT_ERROR}")
if SHAP_IMPORT_ERROR is not None:
    log(f"SHAP 不可用，已自动回退到 Permutation Importance：{SHAP_IMPORT_ERROR}")
log("\n── 真实数据区间 ──")
for column in ["NASDAQ", "VIX", "PE_PCT", "DXY"]:
    series = raw[column].dropna()
    log(f"{column}: {series.index.min().date()} ~ {series.index.max().date()}  ({len(series)} rows)")

# ═══════════════════════════════════════════════════
# 5. XGBoost 训练
# ═══════════════════════════════════════════════════
if XGBClassifier is not None:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
else:
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

proba = model.predict_proba(X_test_s)   # [p_down, p_flat, p_up]
pred  = model.predict(X_test_s)

log("\n── 测试集分类报告 ──")
log(classification_report(y_test, pred, target_names=['Down','Flat','Up']))
cls_accuracy = float(accuracy_score(y_test, pred))
cls_macro_f1 = float(f1_score(y_test, pred, average="macro"))
cls_weighted_f1 = float(f1_score(y_test, pred, average="weighted"))

# ═══════════════════════════════════════════════════
# 6. 仓位构建（基于概率差）
# ═══════════════════════════════════════════════════
p_up   = proba[:, 2]
p_down = proba[:, 0]
p_flat = proba[:, 1]

# 净多头信号 = P(up) - P(down)，再平滑
raw_signal = pd.Series(p_up - p_down, index=test.index)
signal     = raw_signal.rolling(5).mean().fillna(raw_signal)

# 仓位：信号正 → 做多，信号负 → 空仓（不做空）
# 利用置信度分级：信号越强仓位越重，上限1.2倍
def sig2pos(s):
    if s > 0.3:  return min(1.0 + (s-0.3)*0.5, 1.2)
    elif s > 0.1: return 0.8
    elif s > 0.0: return 0.5
    else:         return 0.0

position = signal.apply(sig2pos).shift(1).fillna(0)

# 计算收益
ret_daily  = test['nasdaq_ret_raw'].fillna(0.0)
strat_ret  = position * ret_daily
cost       = position.diff().abs().fillna(0.0) * 0.0005   # 双边万5摩擦
strat_net  = (strat_ret - cost).fillna(0.0)

nav_strat  = (1 + strat_net).cumprod()
nav_bench  = (1 + ret_daily).cumprod()

# ═══════════════════════════════════════════════════
# 7. 绩效指标
# ═══════════════════════════════════════════════════
def metrics(ret: pd.Series, nav: pd.Series, name: str) -> dict:
    ann_r = nav.iloc[-1] ** (252/len(ret)) - 1
    ann_v = ret.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v != 0 else np.nan
    dd = (nav - nav.cummax()) / nav.cummax()
    mdd = dd.min()
    calmar = ann_r / abs(mdd) if mdd != 0 else np.nan
    wr = (ret > 0).mean()
    return {
        "name": name,
        "annual_return": float(ann_r),
        "annual_volatility": float(ann_v),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown": float(mdd),
        "calmar": float(calmar) if not np.isnan(calmar) else None,
        "win_rate": float(wr),
        "total_return": float(nav.iloc[-1] - 1),
    }


def metrics_display(row: dict) -> dict:
    def pct(value: Optional[float]) -> str:
        return f"{value:.2%}" if value is not None else "NA"

    def num(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "NA"
        return f"{value:.{digits}f}"

    return {
        "名称": row["name"],
        "年化收益": pct(row["annual_return"]),
        "年化波动": pct(row["annual_volatility"]),
        "夏普比率": num(row["sharpe"], 3),
        "最大回撤": pct(row["max_drawdown"]),
        "Calmar": num(row["calmar"], 3),
        "胜率": pct(row["win_rate"]),
        "总收益": pct(row["total_return"]),
    }

strategy_metrics = metrics(strat_net, nav_strat, f"{MODEL_LABEL} 多因子策略")
benchmark_metrics = metrics(ret_daily, nav_bench, "NASDAQ基准")
metrics_df = pd.DataFrame([metrics_display(strategy_metrics), metrics_display(benchmark_metrics)]).set_index("名称")

log("\n══════════ 测试集绩效 ══════════")
log(metrics_df.to_string())

# ═══════════════════════════════════════════════════
# 8. 因子重要性
# ═══════════════════════════════════════════════════
if shap is not None:
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test_s)   # shape: (n_samples, n_features, n_classes)

    if isinstance(shap_vals, list):
        shap_abs = np.mean([np.abs(class_values) for class_values in shap_vals], axis=0)
    else:
        shap_arr = np.asarray(shap_vals)
        if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X_test) and shap_arr.shape[1] == len(FACTOR_COLS):
            shap_abs = np.abs(shap_arr).mean(axis=2)
        elif shap_arr.ndim == 3 and shap_arr.shape[1] == len(X_test) and shap_arr.shape[2] == len(FACTOR_COLS):
            shap_abs = np.abs(shap_arr).mean(axis=0)
        elif shap_arr.ndim == 2:
            shap_abs = np.abs(shap_arr)
        else:
            raise ValueError(f"未识别的 SHAP 输出形状: {shap_arr.shape}")

    importance = pd.Series(shap_abs.mean(axis=0), index=FACTOR_COLS).sort_values(ascending=False)
else:
    perm = permutation_importance(
        model,
        X_test_s,
        y_test,
        scoring="accuracy",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.Series(perm.importances_mean, index=FACTOR_COLS).sort_values(ascending=False)

# 模型内置重要性
model_imp = pd.Series(model.feature_importances_, index=FACTOR_COLS).sort_values(ascending=False)

# 因子相关性矩阵（测试集）
corr_matrix = X_test[FACTOR_COLS].corr()

# ═══════════════════════════════════════════════════
# 9. 可视化
# ═══════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor':  '#0d1117',
    'figure.facecolor':'#0d1117',
    'text.color':      '#e6edf3',
    'axes.labelcolor': '#8b949e',
    'axes.edgecolor':  '#30363d',
    'xtick.color':     '#8b949e',
    'ytick.color':     '#8b949e',
    'grid.color':      '#21262d',
    'grid.linewidth':  0.6,
})

C_STRAT = '#58a6ff'
C_BENCH = '#3fb950'
C_NEG   = '#f85149'
C_WARN  = '#ffa657'
C_PURPLE= '#d2a8ff'

fig = plt.figure(figsize=(20, 28))
gs  = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.38,
                        top=0.94, bottom=0.04, left=0.07, right=0.97)

pct_fmt  = FuncFormatter(lambda x, _: f'{x:.0f}%')
nav_fmt  = FuncFormatter(lambda x, _: f'{x:.2f}x')

# ── ① 净值曲线
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(nav_strat.index, nav_strat, color=C_STRAT, lw=2,   label=f'{MODEL_LABEL} Multi-Factor (net of cost)')
ax1.plot(nav_bench.index, nav_bench, color=C_BENCH, lw=1.5, label='NASDAQ Benchmark', alpha=0.85)
ax1.fill_between(nav_strat.index, nav_strat, nav_bench,
                 where=nav_strat >= nav_bench, alpha=0.15, color=C_STRAT)
ax1.fill_between(nav_strat.index, nav_strat, nav_bench,
                 where=nav_strat <  nav_bench, alpha=0.15, color=C_NEG)
ax1.set_title('NAV Curve — Test Set (2021–2024)', fontsize=13, fontweight='bold',
              color='#e6edf3', pad=10)
ax1.yaxis.set_major_formatter(nav_fmt)
ax1.legend(fontsize=10, framealpha=0.2, facecolor='#161b22', edgecolor='#30363d')
ax1.grid(True)

# ── ② 回撤
ax2 = fig.add_subplot(gs[1, :])
dd_s = (nav_strat - nav_strat.cummax()) / nav_strat.cummax() * 100
dd_b = (nav_bench - nav_bench.cummax()) / nav_bench.cummax() * 100
ax2.fill_between(dd_s.index, dd_s, alpha=0.7, color=C_NEG,    label='Strategy Drawdown')
ax2.fill_between(dd_b.index, dd_b, alpha=0.3, color='#8b949e',label='Benchmark Drawdown')
ax2.set_title('Drawdown', fontsize=13, fontweight='bold', color='#e6edf3', pad=10)
ax2.yaxis.set_major_formatter(pct_fmt)
ax2.legend(fontsize=10, framealpha=0.2, facecolor='#161b22', edgecolor='#30363d')
ax2.grid(True)

# ── ③ XGB 预测概率（rolling mean）
ax3 = fig.add_subplot(gs[2, :])
roll = 10
ax3.plot(test.index, pd.Series(p_up,   index=test.index).rolling(roll).mean(),
         color=C_BENCH,  lw=1.2, label='P(Up)')
ax3.plot(test.index, pd.Series(p_flat, index=test.index).rolling(roll).mean(),
         color=C_WARN,   lw=1.2, label='P(Flat)', alpha=0.8)
ax3.plot(test.index, pd.Series(p_down, index=test.index).rolling(roll).mean(),
         color=C_NEG,    lw=1.2, label='P(Down)')
ax3.fill_between(test.index, signal.clip(lower=0), alpha=0.18, color=C_STRAT)
ax3.set_title(f'{MODEL_LABEL} Predicted Probabilities (10d rolling avg)', fontsize=12,
              fontweight='bold', color='#e6edf3', pad=8)
ax3.legend(fontsize=9, framealpha=0.2, facecolor='#161b22', edgecolor='#30363d', ncol=3)
ax3.grid(True)

# ── ④ 因子重要性（Top 15）
ax4 = fig.add_subplot(gs[3, 0])
top_n = 15
top_imp = importance.head(top_n)
colors_bar = [C_STRAT if i < 5 else C_PURPLE if i < 10 else '#8b949e'
              for i in range(top_n)]
bars = ax4.barh(range(top_n), top_imp.values[::-1], color=colors_bar[::-1],
                edgecolor='#30363d', linewidth=0.5)
ax4.set_yticks(range(top_n))
ax4.set_yticklabels([n.replace("nasdaq_","ndq_") for n in top_imp.index[::-1]], fontsize=8)
ax4.set_title(f'{IMPORTANCE_LABEL} Factor Importance (Top 15)', fontsize=11, fontweight='bold',
              color='#e6edf3', pad=8)
ax4.set_xlabel(f'{IMPORTANCE_LABEL} importance', fontsize=9, color='#8b949e')
ax4.grid(True, axis='x', alpha=0.5)
# 图例
patches = [mpatches.Patch(color=C_STRAT,  label='Top 1–5'),
           mpatches.Patch(color=C_PURPLE, label='Top 6–10'),
           mpatches.Patch(color='#8b949e',label='Top 11–15')]
ax4.legend(handles=patches, fontsize=8, framealpha=0.2,
           facecolor='#161b22', edgecolor='#30363d')

# ── ⑤ 模型内置重要性 vs 外部重要性对比（Top 10）
ax5 = fig.add_subplot(gs[3, 1])
top10 = importance.head(10).index
xgb_top10  = model_imp[top10] / model_imp[top10].max()
shap_top10 = top_imp.head(10) / top_imp.head(10).max()

x_pos = np.arange(len(top10))
ax5.bar(x_pos - 0.2, shap_top10.values, 0.35, label=IMPORTANCE_LABEL, color=C_STRAT, alpha=0.85)
ax5.bar(x_pos + 0.2, xgb_top10.values,  0.35, label=f'{MODEL_LABEL} built-in', color=C_WARN, alpha=0.85)
ax5.set_xticks(x_pos)
ax5.set_xticklabels([n.replace('nasdaq_','ndq_') for n in top10], rotation=45,
                    ha='right', fontsize=8)
ax5.set_title(f'{IMPORTANCE_LABEL} vs {MODEL_LABEL} Built-in Importance (Top 10, normalized)', fontsize=10,
              fontweight='bold', color='#e6edf3', pad=8)
ax5.legend(fontsize=9, framealpha=0.2, facecolor='#161b22', edgecolor='#30363d')
ax5.grid(True, axis='y', alpha=0.5)

# ── ⑥ 相关性矩阵（Top 16 因子）
ax6 = fig.add_subplot(gs[4, :])
top16 = importance.head(16).index.tolist()
sub_corr = corr_matrix.loc[top16, top16]

cmap = LinearSegmentedColormap.from_list(
    'rg', ['#f85149', '#161b22', '#3fb950'], N=256)
im = ax6.imshow(sub_corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
ax6.set_xticks(range(len(top16)))
ax6.set_yticks(range(len(top16)))
labels16 = [n.replace('nasdaq_','ndq_') for n in top16]
ax6.set_xticklabels(labels16, rotation=45, ha='right', fontsize=8)
ax6.set_yticklabels(labels16, fontsize=8)
# 标注数值
for i in range(len(top16)):
    for j in range(len(top16)):
        v = sub_corr.values[i, j]
        ax6.text(j, i, f'{v:.2f}', ha='center', va='center',
                 fontsize=6.5, color='white' if abs(v) > 0.5 else '#8b949e')
plt.colorbar(im, ax=ax6, fraction=0.02, pad=0.02)
ax6.set_title(f'Factor Correlation Matrix (Top 16 by {IMPORTANCE_LABEL})', fontsize=11,
              fontweight='bold', color='#e6edf3', pad=8)

# ── ⑦ 绩效指标表
ax7 = fig.add_subplot(gs[5, :])
ax7.axis('off')
cell_text = metrics_df.values.tolist()
table = ax7.table(
    cellText=cell_text,
    rowLabels=list(metrics_df.index),
    colLabels=list(metrics_df.columns),
    cellLoc='center', rowLoc='center', loc='center',
    bbox=[0, -0.1, 1, 1.2]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#30363d')
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor('#21262d')
        cell.set_text_props(color='#8b949e', fontweight='bold')
    elif c == -1:
        cell.set_facecolor('#161b22')
        fc = C_STRAT if r == 1 else C_BENCH
        cell.set_text_props(color=fc, fontweight='bold')
    else:
        cell.set_facecolor('#0d1117' if r%2==0 else '#161b22')
        cell.set_text_props(color='#e6edf3')
ax7.set_title('Performance Metrics — Test Set', fontsize=12, fontweight='bold',
              color='#e6edf3', pad=12, loc='left', x=0.01)

fig.suptitle(
    f'{MODEL_LABEL} Multi-Factor Strategy  |  VIX · PE Percentile · NASDAQ · DXY  '
    '|  Train: 2015–2020   Test: 2021–2024',
    fontsize=14, fontweight='bold', color='#e6edf3', y=0.975)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(REPORT_PATH, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
log(f"\n图表已保存：{REPORT_PATH}")

# ── 额外输出：因子重要性详细表
log(f"\n══════════ {IMPORTANCE_LABEL} 因子重要性排名 ══════════")
imp_table = pd.DataFrame({
    f'{IMPORTANCE_LABEL}_importance': importance,
    f'{MODEL_LABEL}_importance':  model_imp,
    f'{IMPORTANCE_LABEL}_rank': range(1, len(importance)+1),
    f'{MODEL_LABEL}_rank':  model_imp.rank(ascending=False).astype(int)
}).sort_values(f'{IMPORTANCE_LABEL}_importance', ascending=False)
log(imp_table.head(20).to_string())

# ═══════════════════════════════════════════════════
# 10. 标准化输出（JSON/CSV）
# ═══════════════════════════════════════════════════

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")

warnings_list = []
if XGBOOST_IMPORT_ERROR is not None:
    warnings_list.append("XGBoost 不可用，已自动回退到 RandomForest")
if SHAP_IMPORT_ERROR is not None:
    warnings_list.append("SHAP 不可用，已自动回退到 Permutation Importance")

actual_model_used = MODEL_LABEL

# NAV / Drawdown / Returns (test set only)
nav_strat = nav_strat / nav_strat.iloc[0]
nav_bench = nav_bench / nav_bench.iloc[0]
dd_strat_dec = nav_strat / nav_strat.cummax() - 1.0
dd_bench_dec = nav_bench / nav_bench.cummax() - 1.0
excess_nav = nav_strat / nav_bench

nav_df = pd.DataFrame(
    {
        "date": pd.to_datetime(nav_strat.index).strftime("%Y-%m-%d"),
        "strategy_nav": nav_strat.values,
        "benchmark_nav": nav_bench.values,
        "strategy_drawdown": dd_strat_dec.values,
        "benchmark_drawdown": dd_bench_dec.values,
        "strategy_daily_return": pd.to_numeric(strat_net, errors="coerce").values,
        "benchmark_daily_return": pd.to_numeric(ret_daily, errors="coerce").values,
        "excess_nav": excess_nav.values,
        "position": position.reindex(nav_strat.index).fillna(0).values,
    }
)
write_csv(OUTPUT_ROOT / "nav.csv", nav_df)

signals_df = pd.DataFrame(
    {
        "date": pd.to_datetime(test.index).strftime("%Y-%m-%d"),
        "signal_raw": raw_signal.values,
        "signal_smoothed": signal.values,
        "position": position.values,
        "p_down": p_down,
        "p_flat": p_flat,
        "p_up": p_up,
        "pred_label": pd.Series(pred, index=test.index).astype(int).values,
        "true_label": y_test.astype(int).values,
    }
)
write_csv(OUTPUT_ROOT / "signals.csv", signals_df)

importance_method = IMPORTANCE_LABEL
importance_rank = importance.rank(ascending=False, method="first").astype(int)
model_rank = model_imp.rank(ascending=False, method="first").astype(int)
importance_df = pd.DataFrame(
    {
        "feature": importance.index,
        "importance_score": importance.values,
        "importance_method": [importance_method] * len(importance),
        "model_importance": model_imp.reindex(importance.index).values,
        "importance_rank": importance_rank.reindex(importance.index).values,
        "model_rank": model_rank.reindex(importance.index).values,
    }
).sort_values("importance_rank")
write_csv(OUTPUT_ROOT / "features_importance.csv", importance_df)

# Correlation: long table for Top 16 features
top16 = importance.head(16).index.tolist()
sub_corr = corr_matrix.loc[top16, top16]
rows = []
for fx in top16:
    for fy in top16:
        rows.append({"feature_x": fx, "feature_y": fy, "corr": float(sub_corr.loc[fx, fy])})
factor_corr_df = pd.DataFrame(rows)
write_csv(OUTPUT_ROOT / "factor_correlation.csv", factor_corr_df)

# Trades: optional; write empty file with header for now
trades_path = OUTPUT_ROOT / "trades.csv"
trades_header = pd.DataFrame(
    columns=[
        "trade_id",
        "entry_date",
        "exit_date",
        "direction",
        "entry_price",
        "exit_price",
        "holding_days",
        "pnl",
        "pnl_pct",
        "max_favorable_excursion",
        "max_adverse_excursion",
    ]
)
write_csv(trades_path, trades_header)

metrics_payload = {
    "schema_version": SCHEMA_VERSION,
    "columns": [
        "name",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "win_rate",
        "total_return",
    ],
    "rows": [strategy_metrics, benchmark_metrics],
}
write_json(OUTPUT_ROOT / "metrics.json", metrics_payload)

summary_payload = {
    "schema_version": SCHEMA_VERSION,
    "strategy_slug": STRATEGY_SLUG,
    "strategy_name": STRATEGY_NAME,
    "headline": {
        "total_return": strategy_metrics["total_return"],
        "annual_return": strategy_metrics["annual_return"],
        "annual_volatility": strategy_metrics["annual_volatility"],
        "sharpe": strategy_metrics["sharpe"],
        "max_drawdown": strategy_metrics["max_drawdown"],
        "calmar": strategy_metrics["calmar"],
        "win_rate": strategy_metrics["win_rate"],
    },
    "benchmark": {
        "name": "NASDAQ",
        "total_return": benchmark_metrics["total_return"],
        "annual_return": benchmark_metrics["annual_return"],
        "annual_volatility": benchmark_metrics["annual_volatility"],
        "sharpe": benchmark_metrics["sharpe"],
        "max_drawdown": benchmark_metrics["max_drawdown"],
        "calmar": benchmark_metrics["calmar"],
        "win_rate": benchmark_metrics["win_rate"],
    },
    "classification": {
        "accuracy": cls_accuracy,
        "macro_f1": cls_macro_f1,
        "weighted_f1": cls_weighted_f1,
    },
    "top_features": [
        {"name": row["feature"], "score": float(row["importance_score"]), "method": row["importance_method"]}
        for _, row in importance_df.head(10).iterrows()
    ],
}
write_json(OUTPUT_ROOT / "summary.json", summary_payload)

config_payload = {
    "schema_version": SCHEMA_VERSION,
    "editable": True,
    "groups": [
        {
            "group_key": "data",
            "group_name": "数据区间",
            "items": [
                {"key": "train_start", "label": "训练开始", "type": "date", "value": TRAIN_START},
                {"key": "train_end", "label": "训练结束", "type": "date", "value": TRAIN_END},
                {"key": "test_start", "label": "测试开始", "type": "date", "value": TEST_START},
                {"key": "test_end", "label": "测试结束", "type": "date", "value": TEST_END},
            ],
        },
        {
            "group_key": "labeling",
            "group_name": "标签设置",
            "items": [
                {"key": "horizon", "label": "预测周期", "type": "int", "value": int(HORIZON), "min": 1, "max": 60, "step": 1},
                {"key": "threshold", "label": "涨跌阈值", "type": "float", "value": float(THRESH), "min": 0.0, "max": 0.05, "step": 0.001},
                {"key": "transaction_cost", "label": "手续费", "type": "float", "value": 0.0005, "min": 0.0, "max": 0.01, "step": 0.0001},
            ],
        },
        {
            "group_key": "model",
            "group_name": "模型参数",
            "items": [
                {"key": "model_name", "label": "模型", "type": "string", "value": "XGBoost", "readonly": False},
                {"key": "actual_model_used", "label": "实际使用", "type": "string", "value": actual_model_used, "readonly": True},
            ],
        },
    ],
}
write_json(OUTPUT_ROOT / "config.json", config_payload)

manifest_payload = {
    "schema_version": SCHEMA_VERSION,
    "strategy_slug": STRATEGY_SLUG,
    "strategy_name": STRATEGY_NAME,
    "script_path": f"strategy/{STRATEGY_SLUG}.py",
    "status": "success",
    "run_id": RUN_ID,
    "generated_at": GENERATED_AT,
    "description": "基于 VIX、PE 分位、NASDAQ、DXY 的多因子分类策略（示例策略）",
    "tags": ["美股", "多因子", "机器学习"],
    "model": {
        "name": "XGBoost",
        "fallback_name": "RandomForest",
        "actual_model_used": actual_model_used,
    },
    "date_range": {
        "data_start": raw.index.min().strftime("%Y-%m-%d"),
        "data_end": raw.index.max().strftime("%Y-%m-%d"),
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
    },
    "artifacts": {
        "summary": "summary.json",
        "metrics": "metrics.json",
        "config": "config.json",
        "nav": "nav.csv",
        "signals": "signals.csv",
        "features_importance": "features_importance.csv",
        "factor_correlation": "factor_correlation.csv",
        "trades": "trades.csv",
        "logs": "logs.txt",
        "report_image": "artifacts/report.png",
    },
    "capabilities": {
        "has_nav": True,
        "has_drawdown": True,
        "has_signals": True,
        "has_trades": False,
        "has_feature_importance": True,
        "has_factor_correlation": True,
        "has_param_panel": True,
        "supports_rerun": True,
    },
    "warnings": warnings_list,
    "error": "",
}
write_json(OUTPUT_ROOT / "manifest.json", manifest_payload)

log(f"\n标准化结果已输出到：{OUTPUT_ROOT}")
