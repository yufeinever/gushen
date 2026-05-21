# AKShare 数据接口地图

日期：2026-05-21

## 结论

第一阶段以 AKShare 作为主数据源。AKShare 官方股票数据文档覆盖了 A 股实时行情、历史行情、风险警示板、停复牌、公告、财务报表、板块、龙虎榜、资金流向、涨跌停池等大量接口，足够支撑我们先做成交额前 100 股票池、特征、Agent 观察和回测验证。

当前要解决的不是“AKShare 不够”，而是：

- 选对接口。
- 做接口健康检查。
- 做缓存和降级。
- 避免为了小验证一次性拉取不必要的全量分页数据。

## 第一阶段接口

| 数据 | AKShare 函数 | 用途 | 当前策略 |
| --- | --- | --- | --- |
| A 股代码和名称 | `stock_info_a_code_name` | 建股票主表 | 必接 |
| 交易日历 | `tool_trade_date_hist_sina` | 判断交易日、回测日历 | 必接 |
| 个股历史日线 | `stock_zh_a_hist` | 日线行情、成交额、复权 | 必接 |
| A 股实时行情 | `stock_zh_a_spot_em` | 当日成交额 Top N | 收盘后优先用，失败则缓存/降级 |
| 风险警示板 | `stock_zh_a_st_em` | ST/风险警示过滤 | 必接，但允许缓存 |
| 停复牌 | `stock_tfp_em` / 停复牌相关接口 | 停牌过滤 | 必接 |
| 涨跌停池 | `stock_zt_pool_em` / `stock_zt_pool_dtgc_em` | 涨跌停状态 | 第二步接入 |
| 公告 | 巨潮/沪深京公告相关接口 | EventAnalystAgent | 第二步接入 |

## 验证顺序

1. 跑 `gushen-akshare-doctor` 检查基础接口。
2. 先接股票主表和交易日历。
3. 用 `stock_zh_a_hist` 拉少量股票最近 6 个月日线，验证字段和复权。
4. 再接 `stock_zh_a_spot_em` 生成当日成交额 Top 30。
5. 如果实时接口短时失败，用最近一个可用交易日历史数据或缓存继续验证流程。
6. 最后扩展到 Top 100 和 3 年历史。

## 本地健康检查结果

2026-05-21 本地环境测试：

- `stock_info_a_code_name`：成功，返回 5519 行。
- `tool_trade_date_hist_sina`：成功，返回 8797 行。
- `stock_zh_a_hist`：成功，单股日线返回 89 行。
- `stock_zh_a_st_em`：当前环境失败，错误集中在东方财富接口代理连接。
- `stock_zh_a_spot_em`：当前环境失败，错误集中在东方财富接口代理连接。

这说明 AKShare 主体可用，当前卡点是部分东方财富接口在本地代理环境下连接不稳定。后续优先做缓存、重试和替代接口，而不是更换主数据源。

## 官方文档

- AKShare 股票数据：https://akshare.akfamily.xyz/data/stock/stock.html
