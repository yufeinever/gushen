# AI Agent 底座选型

日期：2026-05-21

## 结论

如果项目要更偏 AI Agent，建议采用“双底座”思路：

- 决策与复盘层复刻 TradingAgents 的多角色 Agent 架构。
- 研究与迭代层借鉴 Microsoft RD-Agent 的自动因子、模型和实验循环。

也就是说，我们不只是让 AI 写一段股票分析，而是让它形成一套可审计的投研组织：

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
- A 股需要接入 AKShare、Tushare、交易日历、涨跌停、T+1、停牌、ST 状态。
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

- 需要把实验目标限定在 A 股成交额前 100 股票池。
- 自动生成的因子必须经过代码审查和回测验证。
- 不允许直接把自动生成的策略进入实盘。

### 3. FinRobot

项目地址：https://github.com/AI4Finance-Foundation/FinRobot

适合复刻的部分：

- 金融分析 Agent 平台。
- 自动生成权益研究报告。
- 多 Agent 处理财务、估值、风险和报告生成。
- Apache-2.0 许可证。

更适合作为：

- 个股研究报告 Agent 的参考。
- 公告和财报分析 Agent 的参考。
- 中长线基本面解释层的参考。

不建议作为第一主线的原因：

- 第一阶段我们更重视 A 股高流动性股票池的每日量价闭环。
- FinRobot 的默认数据源和报告范式更偏美股/海外权益研究。

### 4. AI Hedge Fund

项目地址：https://github.com/virattt/ai-hedge-fund

适合复刻的部分：

- 投资大师风格 Agent。
- Fundamentals、Technicals、Sentiment、Risk Manager、Portfolio Manager 的组合。
- 命令行和 Web 两种形态。
- MIT 许可证。

定位：

- 很适合学习“多 Agent 如何组织观点”。
- 不适合作为严肃交易主底座，因为风格化 Agent 容易变成叙事强、验证弱。

### 5. OpenBB / FinGPT

OpenBB 项目地址：https://github.com/OpenBB-finance/OpenBB

FinGPT 项目地址：https://github.com/AI4Finance-Foundation/FinGPT

定位：

- OpenBB 更像金融数据和分析工作台，可作为数据/工具层参考。
- FinGPT 更偏金融大模型和语料/模型生态，可作为文本模型资源参考。

## 我们的 Agent 架构草案

### Agent 角色

- `UniverseAgent`：每日生成成交额前 100 股票池，剔除停牌、ST、退市风险。
- `DataQualityAgent`：检查缺失、异常、复权、交易日对齐。
- `FeatureAgent`：生成趋势、量价、波动、流动性、涨跌停距离特征。
- `FactorResearchAgent`：提出因子假设并交给回测验证，参考 RD-Agent。
- `TechnicalAnalystAgent`：解释技术结构和短期强弱。
- `EventAnalystAgent`：读取公告、新闻和财报摘要，标记事件风险。
- `BullResearcherAgent`：提出看多理由。
- `BearResearcherAgent`：提出看空理由。
- `RiskManagerAgent`：检查仓位、回撤、涨跌停、T+1、黑名单。
- `PortfolioManagerAgent`：只输出观察、研究、模拟盘动作。
- `ReviewAgent`：每日和每周复盘，记录错误和规则偏离。

### Agent 输出约束

所有 Agent 输出必须结构化：

```json
{
  "date": "2026-05-21",
  "code": "600000.SH",
  "agent": "RiskManagerAgent",
  "verdict": "avoid",
  "reasons": [],
  "supporting_data": [],
  "risks": [],
  "invalid_condition": "",
  "confidence_note": ""
}
```

### 行动枚举

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

## 最小复刻目标

先做一个 `GushenAgents` 原型：

1. 输入每日成交额前 100 股票池和特征表。
2. 技术分析 Agent 给出结构化解释。
3. 事件 Agent 标记公告和新闻风险。
4. 多空 Agent 分别给出正反观点。
5. 风控 Agent 一票否决高风险标的。
6. 组合 Agent 输出 5 到 10 个观察候选。
7. 复盘 Agent 在次日/一周后评估候选表现。

## 最终建议

- 不直接 fork 任一 Agent 项目。
- 先复刻 TradingAgents 的角色组织和决策流程。
- 用 RD-Agent 的思路做自动因子和模型迭代。
- 用 FinRobot 借鉴报告生成。
- 用 Qlib-like 数据和回测层承接 Agent 输出。
- 用 A 股规则和风控政策把 Agent 关进笼子里。
