# AI Agent 底座选型

日期：2026-05-21

## 结论

如果项目要更偏 AI Agent，建议采用“双底座”思路：

- 决策与复盘层复刻 TradingAgents 的多角色 Agent 架构。
- 研究与迭代层借鉴 Microsoft RD-Agent 的自动因子、模型和实验循环。

我们不是让 AI 写一段股票分析，而是让它形成一套可审计的投研组织：

```text
数据工程 Agent -> 特征/因子 Agent -> 技术分析 Agent -> 基本面/公告 Agent
-> 多空辩论 Agent -> 风控 Agent -> 组合 Agent -> 复盘 Agent
```

第一阶段仍然不让 Agent 直接下单。Agent 的产物是观察、研究、模拟盘建议、风险解释和复盘结论。

## 推荐优先级

### 1. TradingAgents

项目地址：https://github.com/TauricResearch/TradingAgents

适合复刻的部分：

- 多 Agent 金融交易框架。
- 分析师团队、研究员团队、交易员、风控、组合经理的角色拆分。
- 多 Agent 辩论和结构化决策。
- 决策日志、检查点恢复、CLI 交互。
- Apache-2.0 许可证，适合借鉴架构。

需要改造的部分：

- 默认更偏美股数据和海外信息源。
- A 股需要接入 AKShare、交易日历、涨跌停、T+1、停牌、ST 状态。
- 输出不能直接是交易指令，第一阶段只能落到观察清单和模拟盘。

### 2. RD-Agent

项目地址：https://github.com/microsoft/RD-Agent

适合复刻的部分：

- 自动提出研究假设。
- 自动实现因子和模型。
- 自动运行实验并根据反馈迭代。
- 和 Qlib 的量化研究场景结合紧密。
- MIT 许可证，适合参考和集成。

需要改造的部分：

- 实验目标限定在 A 股成交额前 100 股票池。
- 自动生成的因子必须经过代码审查和回测验证。
- 不允许直接把自动生成的策略进入实盘。

## 我们的 Agent 架构草案

- `UniverseAgent`：每日生成成交额前 100 股票池，剔除停牌、ST、退市风险。
- `DataQualityAgent`：检查缺失、异常、复权、交易日对齐。
- `FeatureAgent`：生成趋势、量价、波动、流动性、涨跌停距离特征。
- `FactorResearchAgent`：提出因子假设并交给回测验证，参考 RD-Agent。
- `TechnicalAnalystAgent`：解释技术结构和短期强弱。
- `EventAnalystAgent`：读取公告、新闻和财报摘要，标记事件风险。
- `BullResearcherAgent`：提出看多理由。
- `BearResearcherAgent`：提出看空理由。
- `ResearchManagerAgent`：综合多空辩论，形成研究结论。
- `TraderAgent`：把研究结论转成模拟交易计划。
- `RiskManagerAgent`：检查仓位、回撤、涨跌停、T+1、黑名单。
- `PortfolioManagerAgent`：只输出观察、研究、模拟盘动作。
- `ReviewAgent`：每日和每周复盘，记录错误和规则偏离。

## 行动约束

第一阶段只允许：

- `observe`
- `research`
- `paper_trade`
- `avoid`

暂不允许：

- `buy`
- `sell`
- `increase_position`
- `decrease_position`

## 最终建议

- 不直接 fork 任一 Agent 项目。
- 把 TradingAgents 拉到本地作为参考源。
- 复刻 TradingAgents 的角色组织、结构化输出、辩论流程和记忆日志。
- 用 RD-Agent 的思路做自动因子和模型迭代。
- 用 Qlib-like 数据和回测层承接 Agent 输出。
- 用 A 股规则和风控政策约束 Agent。
