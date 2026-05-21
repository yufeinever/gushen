# TradingAgents 本地参考与改造计划

日期：2026-05-21

## 当前状态

已将 TradingAgents 拉取到本地：

```text
external/TradingAgents
```

`external/` 已加入 `.gitignore`。上游源码只作为本地参考，不直接提交进本项目，避免把第三方项目和我们的代码混在一起。

## TradingAgents 核心机制

TradingAgents 的核心不是单个提示词，而是一张 Agent 工作流图：

```text
Analyst Team
  Market / Technical Analyst
  Sentiment Analyst
  News Analyst
  Fundamentals Analyst
-> Bull Researcher
-> Bear Researcher
-> Research Manager
-> Trader
-> Aggressive Risk Analyst
-> Conservative Risk Analyst
-> Neutral Risk Analyst
-> Portfolio Manager
```

关键实现位置：

- `tradingagents/graph/setup.py`：LangGraph 状态图，定义节点和流转。
- `tradingagents/agents/schemas.py`：Research Manager、Trader、Portfolio Manager 的结构化输出。
- `tradingagents/graph/trading_graph.py`：主编排、运行、日志、记忆。
- `tradingagents/agents/utils/memory.py`：追加式决策记忆和事后复盘。
- `tradingagents/graph/conditional_logic.py`：多空辩论和风控辩论的轮次控制。

## 我们要复用的逻辑

### 1. 工作流图

先不直接引入 LangGraph。我们先在 `gushen` 里实现一个轻量顺序图，等流程稳定后再决定是否接 LangGraph。

目标流程：

```text
Universe
-> Technical Analyst
-> Event Analyst
-> Bull Researcher
-> Bear Researcher
-> Research Manager
-> Trader
-> Risk Debate
-> Portfolio Manager
-> Memory / Review
```

### 2. 结构化输出

TradingAgents 只给关键决策角色加结构化输出，这是对的。我们也采用这个分层：

- Analyst 可以输出报告和证据。
- Research Manager 必须输出结构化研究结论。
- Trader 必须输出结构化模拟交易计划。
- Portfolio Manager 必须输出结构化最终动作。

### 3. 记忆日志

TradingAgents 用 append-only markdown 记录每次决策，并在后续运行时注入历史经验。

我们要改造成：

- SQLite 记录结构化决策。
- Markdown 生成可读复盘。
- 每个候选股票保留入池原因、AI 结论、模拟交易计划、后验收益和复盘。

### 4. A 股数据层替换

TradingAgents 默认偏海外数据源。我们替换为：

- AKShare 行情。
- AKShare 公告、财务、涨跌停、停复牌。
- 本地 SQLite 缓存。
- A 股交易规则：T+1、涨跌停、ST、停牌、印花税、手续费。

## 下一步迁移任务

1. 已完成：在 `src/gushen/agent_schemas.py` 中实现 A 股版结构化输出。
2. 已完成：增加 `ResearchManagerAgent` 和 `TraderAgent`。
3. 已完成：增加激进、保守、中性三种风险视角。
4. 待完成：增加 `MemoryLog`：SQLite + Markdown 双记录。
5. 待完成：把 `Top100 + 收盘价 < 50` 的候选结果喂给这套 Agent 流程。
6. 待完成：接入 LLM，让 Agent 真正从提示词和结构化上下文中生成判断。

## 当前已落地代码

- `src/gushen/agent_schemas.py`：A 股版结构化输出。
- `src/gushen/agents.py`：TradingAgents 风格的本地规则版流程。
- `tests/test_agents.py`：覆盖高风险否决和干净候选完整通过。

当前流程：

```text
Universe
-> Technical Analyst
-> Event Analyst
-> Bull Researcher
-> Bear Researcher
-> Research Manager
-> Trader
-> Aggressive Risk
-> Conservative Risk
-> Neutral Risk
-> Risk Manager
-> Portfolio Manager
```

## 2026-05-21 进展

已将 `Top100 + 收盘价 < 50` 候选接入完整 Agent pipeline。

2026-05-20 数据运行结果：

- 候选 30 只。
- `paper_trade` 8 只。
- `research` 14 只。
- `avoid` 8 只。

注意：当前 Agent pipeline 主要读取股票上下文和风险状态，还没有把规则分数作为硬门槛。下一步需要把初始规则分数纳入 Agent 上下文，防止低分候选在后续 Agent 中被误放行。

## 不直接照搬的部分

- 不直接使用海外数据接口。
- 不直接输出 Buy/Sell 实盘指令。
- 不直接使用美股 benchmark 和 SPY alpha 逻辑。
- 不直接使用 Reddit/StockTwits 情绪源。
- 不把上游代码整包复制到 `src/gushen`。
