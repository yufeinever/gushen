# 数据源接口索引

本文件记录 A股研究系统可用的数据源接口类别。详细函数名会随 AKShare 版本变化，开发时应以本地 `DataSourceDoctor` 和 `dir(akshare)` 探测结果为准。

## 接口类别

| 类别 | 主要来源 | 用途 |
| --- | --- | --- |
| 基础股票池 | AKShare、交易所、东方财富 | A股代码、名称、交易日、Top100 成交额股票池 |
| 行情与技术 | 东方财富、AKShare、交易所 | 日线、实时行情、成交额、换手、复权历史 |
| 交易限制 | 东方财富、交易所 | ST、停复牌、涨跌停、风险警示 |
| 公告与新闻 | 东方财富、巨潮资讯、交易所、web 搜索 | 个股公告、新闻、事件标签和事件摘要 |
| 财务与估值 | 东方财富、同花顺、巨潮资讯 | PE/PB、市值、财报、资产负债表、利润表、现金流 |
| 行业与概念 | 东方财富 EM、同花顺 THS、申万 SW、巨潮 CNInfo、新浪 | 行业/概念归属、板块强弱、板块成份、行业指数 |
| 资金流 | 东方财富、同花顺、沪深港通、交易所 | 主力资金、板块资金、北向资金、融资融券、龙虎榜 |
| 宏观与流动性 | AKShare、东方财富、银行间/交易所数据 | 利率、汇率、LPR、SHIBOR、PMI、波动率指数 |
| 论坛与叙事 | 股吧、雪球、新闻站点、web 搜索 | 市场叙事、反向观点、拥挤度和可信度判断 |

## 当前多源方向

- `SectorThemeAgent`：优先尝试东方财富 EM 成分映射；若不可用，使用同花顺 THS 行业/概念摘要、THS 行业/概念资金流、申万 SW 行业分类、新浪板块明细、巨潮 CNInfo 行业分类等替代源，状态标记为 `partial`；最后才回退本地价量 fallback。
- `FundFlowAgent`：优先个股主力资金，其次板块资金、北向、融资融券、龙虎榜；不可用时明确标记 `partial` 或 `fallback`。
- `DataSourceDoctor`：负责记录接口健康状态，不把失败接口伪装成真实数据。

## 相关文件

- 数据源诊断实现：`src/gushen/data_source_doctor.py`
- 数据集构建实现：`src/gushen/tradingagents_dataset.py`
- 数据质量闸门：`src/gushen/data_quality.py`
- 看板展示：`src/gushen/dashboard.py`
- AKShare 初始地图：`docs/AKSHARE_SOURCE_MAP.md`
