# AI 接口与 Agent 运行方式

日期：2026-05-21

## 是否需要 Codex subagent

交易系统里的 Agent 不等于 Codex subagent。

我们当前的 Agent 是业务流程节点：

- Technical Analyst
- Event Analyst
- Bull Researcher
- Bear Researcher
- Research Manager
- Trader
- Risk Debate
- Portfolio Manager

这些节点应该在 `gushen` 代码里运行，并记录到本地 SQLite 和报告中。Codex subagent 适合开发时并行读代码、修模块，不适合作为交易系统运行时依赖。

## 当前运行方式

当前已经跑通本地规则版完整流程：

```text
Top100 成交额股票池
-> 收盘价 < 50
-> 规则打分
-> TradingAgents-style Agent pipeline
-> SQLite + CSV 报告
```

2026-05-20 结果：

- 候选：30 只。
- `paper_trade`：8 只。
- `research`：14 只。
- `avoid`：8 只。

报告：

```text
reports/generated/top100_under50_agent_pipeline_2026-05-20.csv
```

## 真正接 AI 需要什么

需要你提供 OpenAI-compatible 接口配置：

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
GUSHEN_LLM_MODEL=...
```

建议至少两档模型：

- `research`：高推理模型，用于 Research Manager、Portfolio Manager。
- `extraction`：低成本模型，用于公告、新闻和财务字段抽取。

## 为什么先不直接调用 AI

- 现在还没有你的 API key 和 base URL。
- 交易系统不能依赖 Codex 会话能力临时做判断。
- 必须先有结构化输入、输出 schema、日志和回测，再接 LLM。
- 每次 AI 输出都要能追溯、缓存、复盘和成本统计。

## 下一步

1. 把初始规则分数作为 Agent pipeline 的上下文，避免低分候选绕过打分逻辑。
2. 实现 LLM adapter，支持 OpenAI-compatible 接口。
3. 让 Research Manager、Trader、Portfolio Manager 从规则版切换为 LLM + schema 输出。
4. 所有 AI 输出写入 SQLite。
5. 开始做 1/3/5 日模拟回测。
