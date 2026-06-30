# AGENTS.md

## Project Instructions

- 从现在开始，本项目默认就在当前 Kylin 设备上开发、测试、提交和推送；本地 Windows 目录 `C:\\Users\\86150\\Documents\\gushen` 仅作迁移残留参考，不作为最新项目状态来源，不直接更新项目代码。
- 国内数据源默认直连，不主动走代理。包括但不限于 AKShare 调用的东方财富、新浪、巨潮资讯、交易所等国内站点；只有确认直连不通且目标确实需要代理时，才临时设置代理。
- 访问 GitHub、OpenAI、海外文档或其他外网资源时，如果网络不通，可以使用本地代理，例如 `http://127.0.0.1:10808`。
- 需要 AI 深度分析时，使用本地 `.env` 中的 OpenAI-compatible 配置：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`GUSHEN_LLM_MODEL`。`.env` 是私有文件，不能提交，不能在回复或日志中明文输出 API key。
- 股票分析必须先说明数据充分性。数据不足、字段缺失、样本太短、未接公告/涨跌停/停牌/ST/回测时，不能输出“推荐”“买入”“卖出”或暗示实盘动作；只能标记为数据不足、观察、研究或模拟验证，并优先说明还缺哪些数据。
- 盘中快照、实时成交额榜、实时资金流、即时大单等是重要数据源，优先用于盘中监控、执行确认、风险预警和观察信号；不能回填历史交易日，也不能伪装成收盘后的历史 Top100 选股池。
- A股数据接口按类别维护：基础股票池、行情与技术、交易限制、公告新闻、财务估值、行业概念、资金流、宏观流动性、论坛叙事。AGENTS.md 只保留分类和规则，详细接口索引见 `docs/DATA_SOURCE_INTERFACE_INDEX.md`，不要把详细接口清单写进本文件。
- Backtest baseline comparison must include data sufficiency first, the strategy return, the stock anchor-window-low hold baseline, the aligned SSE Composite index hold return using the same entry/exit dates, and excess return versus that index. Do not describe a strategy as effective when it only beats the stock baseline but underperforms the aligned index benchmark.
- Guided factor backtests must prioritize the stock pool in `docs/GUIDED_FACTOR_BACKTEST_STOCKS.md`; factor screening is per-stock, because each stock can have a different effective factor set. Do not reuse one universal factor list without per-stock evidence.
- Guided strategy-library searches must use only the two-year research window, score factors per stock on a training segment, choose factor/threshold/holding candidates on a separate validation segment, and report strategy returns only from the final holdout segment.
- Guided factor trade execution now uses backtesting.py; keep signal generation separate from the execution engine.
- Windows PowerShell 通过 SSH 给远端传脚本时，中文/emoji 字符串可能被转码破坏；筛选或写入中文必须用 Unicode escape、远端原始值、节点序号或 server/port，不能直接在本地命令里写中文条件。
- 远端更新前后都要查目标行数。
- 项目改动后要提交。
