# [OPEN] indicator-fetch-failures

## 现象

- `us_stock_indicator_pipeline.py` 运行时，部分指标成功，部分指标失败。
- 失败主要集中在三类：
  - `FRED` 相关指标超时
  - `yfinance` 相关指标限流或 curl 错误
  - 个别网页型源在当前环境可访问性不稳定

## 复现命令

```bash
python3 us_stock_indicator_pipeline.py
```

## 初始假设

1. `FRED` 失败不只是“接口慢”，而是当前请求方式、响应格式或 TLS/网络链路存在系统性问题。
2. `yfinance` 失败的根因是当前环境被 Yahoo 限流，而不是 ticker 本身无效。
3. `AKShare` 替代源对美股 ETF/宏观数据更稳定，可作为部分失败指标的兜底方案。
4. 失败指标里有一部分可以通过“已有本地 CSV + 公共静态 CSV + AKShare”组合补齐，而不必继续依赖 `FRED/yfinance`。
5. 当前脚本缺少足够细的抓数阶段证据，导致无法快速区分“源不可用”“解析失败”“限流”“超时”。

## 计划

1. 对抓数入口增加最小化调试插桩，记录数据源、请求目标、失败类型与 ticker/series 信息。
2. 重新运行脚本，收集运行时证据。
3. 基于证据替换最容易失败的公开数据源。
4. 再次运行并比对修复前后结果。

## 运行时证据

- `FRED` 相关失败统一表现为 `ReadTimeout`，已确认不是单个 series id 配置错误。
- `CNH=X` 在 `yfinance` 通道下返回空数据，失败类型为 `ValueError: No price data returned for CNH=X`。
- `AKShare` 路径对美股 ETF、技术面和部分宏观指标稳定。
- `M2SL` 可通过 GitHub 上的公开镜像 CSV 正常获取。
- `USD/CNY` 可通过 `Frankfurter` 公开接口稳定返回时间序列，可作为 `USDCNH` 的环境代理。

## 已实施修复

1. 为抓数入口补充了调试上报，记录源、标的、异常类型和失败指标。
2. 将 `M2 货币供应量同比` 从 `FRED` 切换为公开 GitHub 镜像 CSV。
3. 将 `人民币汇率（USDCNH）` 从 `yfinance: CNH=X` 切换为 `Frankfurter USD/CNY` 代理源。

## 修复前后对比

- 修复前：
  - `M2 货币供应量同比` 失败，原因为 `FRED ReadTimeout`
  - `人民币汇率（USDCNH）` 失败，原因为 `CNH=X` 空数据
- 修复后：
  - `M2 货币供应量同比` 获取成功
  - `人民币汇率（USDCNH）` 获取成功

## 仍未解决

- `美联储资产负债表规模`
- `SOFR / 隔夜利率`
- `通胀预期（5Y5Y / 密歇根）`
- `房租 OER`
- `信用利差（IG / HY spread）`
- `芝加哥联储 NFCI`

以上剩余失败项仍然都指向同一类根因：当前环境对 `fred.stlouisfed.org` 访问持续超时，需要逐项替换为非 FRED 的公开源。
