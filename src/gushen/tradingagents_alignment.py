from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlignmentRow:
    upstream_component: str
    upstream_behavior: str
    gushen_equivalent: str
    a_share_adaptation: str
    status: str
    score: int
    next_action: str


ALIGNMENT_ROWS: tuple[AlignmentRow, ...] = (
    AlignmentRow(
        "Market Analyst",
        "通过工具节点读取价格和技术指标，形成市场报告。",
        "已有 market_technical.csv、TechnicalAnalystAgent 和看板打分。",
        "成交额 Top100 股票池、T+1 交易约束、涨跌停和停牌状态。",
        "partial",
        58,
        "补充更多技术指标和相对指数/行业强弱。",
    ),
    AlignmentRow(
        "Sentiment Analyst",
        "通过工具读取新闻/社交流，输出情绪报告。",
        "已有 SentimentNarrativeAgent 名称，但还没有接入 A股论坛原始流。",
        "东方财富股吧、雪球、可靠网页搜索，并对来源可信度打分。",
        "weak",
        25,
        "构建 A股版 SentimentNarrativeAgent 原始数据流和反向观点校验。",
    ),
    AlignmentRow(
        "News Analyst",
        "读取个股新闻、全球新闻和重大事件。",
        "events.csv 已读取 stock_news_em 和公告标题。",
        "交易所公告、政策新闻、公告正文抽取和事件影响分类。",
        "partial",
        48,
        "抽取公告正文，并按利好/利空/中性/不确定分类。",
    ),
    AlignmentRow(
        "Fundamentals Analyst",
        "读取基本面、资产负债表、现金流和利润表工具。",
        "fundamentals.csv 目前主要是估值 fallback。",
        "A股财报字段、股权质押、限售解禁、审计质量和盈利质量。",
        "weak",
        30,
        "接入财报数据集和财务质量标记。",
    ),
    AlignmentRow(
        "Bull Researcher",
        "基于分析师报告提出看多论证。",
        "BullResearcherAgent 已存在，但目前是浅层规则。",
        "A股题材、政策、资金流和事件催化。",
        "partial",
        45,
        "把各分析师产物汇总成结构化看多论点。",
    ),
    AlignmentRow(
        "Bear Researcher",
        "基于分析师报告提出看空/风险论证。",
        "BearResearcherAgent 已有波动率、涨跌停、动量风险。",
        "监管、估值、情绪拥挤、解禁和流动性风险。",
        "partial",
        48,
        "扩大风险来源，并要求每条风险带数据依据。",
    ),
    AlignmentRow(
        "Research Manager",
        "用更强模型综合多空辩论。",
        "ResearchManagerAgent 已存在，但目前是规则综合。",
        "只有通过数据质量闸门后，才调用私有 LLM 深度综合。",
        "partial",
        50,
        "把规则综合升级为受数据闸门约束的结构化 LLM 调用。",
    ),
    AlignmentRow(
        "Trader",
        "把研究结论转换成交易计划。",
        "TraderAgent 已能输出模拟交易计划。",
        "A股 T+1、集合竞价/开盘/VWAP 模拟、禁止日内同日退出。",
        "partial",
        55,
        "把建仓/出仓规则接到更接近执行的回测结果上。",
    ),
    AlignmentRow(
        "Risk Debate",
        "激进、保守、中性三个风险角色进行辩论。",
        "三个风险 Agent 已存在，但目前是单轮规则判断。",
        "A股回撤、流动性、拥挤度、宏观和题材风险辩论。",
        "partial",
        50,
        "持久化每个风险角色观点，并至少执行一轮显式辩论。",
    ),
    AlignmentRow(
        "Portfolio Manager",
        "风险讨论后给出最终组合决策。",
        "PortfolioManagerAgent 已存在，但只服务模拟动作。",
        "资金账户、已有持仓、行业集中度和组合暴露。",
        "weak",
        38,
        "接入账户/持仓状态，并做组合暴露检查。",
    ),
    AlignmentRow(
        "Tool Nodes",
        "LangGraph Agent 会反复调用工具，直到报告完整。",
        "当前流程是先生成 CSV，再跑确定性函数。",
        "国内 AKShare/网页搜索工具注册表，国内数据默认不走代理。",
        "missing",
        15,
        "构建本地工具注册表和 Agent 工具调用循环。",
    ),
    AlignmentRow(
        "Memory / Reflection",
        "追加式记录决策，并在结果出来后做复盘反思。",
        "还没有把本地记忆日志接入决策。",
        "跟踪模拟交易结果、错误归因和可复用的 A股经验。",
        "missing",
        10,
        "实现 MemoryReviewAgent 和追加式 markdown 复盘日志。",
    ),
    AlignmentRow(
        "Data Quality Gate",
        "原版隐式依赖工具数据可用性。",
        "已新增 DataQualityAgent，在动作前评估数据完整性。",
        "核心数据不完整时硬停止；关键 A股语境缺失时只能研究观察。",
        "partial",
        60,
        "让 Trader/Portfolio 的每条路径都消费数据闸门结果。",
    ),
    AlignmentRow(
        "Macro / Policy Context",
        "原版可通过全球新闻补宏观，但没有 A股宏观状态 Agent。",
        "MacroRegimeAgent 已使用利率、汇率、LPR、SHIBOR、PMI 和 QVIX。",
        "国内政策、流动性和风格状态修正。",
        "adapted",
        65,
        "补政策日历，以及行业/风格敏感度。",
    ),
    AlignmentRow(
        "Sector / Theme Context",
        "不是原版核心组件。",
        "已生成 sector_themes.csv；接入 THS 板块强弱，并用新浪行业成分缓存补 Top100 个股行业映射，仍标记 partial。",
        "A股行业、概念和题材强弱是核心语境。",
        "partial",
        55,
        "继续补概念成分映射和跨源行业口径统一，让 SectorThemeAgent 从 partial 升到 ok。",
    ),
    AlignmentRow(
        "Fund Flow Context",
        "不是原版核心组件。",
        "已生成 fund_flows.csv；接入 stock_individual_fund_flow 个股历史主力资金缓存，并叠加 HSGT、融资融券和龙虎榜。",
        "主力资金、北向、融资融券和龙虎榜是 A股核心语境。",
        "partial",
        45,
        "提高 Top100 个股资金流覆盖率，并补充大单/板块资金流对缺失个股的替代映射。",
    ),
)


def build_alignment_payload() -> dict[str, object]:
    rows = [asdict(row) for row in ALIGNMENT_ROWS]
    average = round(sum(row.score for row in ALIGNMENT_ROWS) / len(ALIGNMENT_ROWS), 2)
    status_counts: dict[str, int] = {}
    for row in ALIGNMENT_ROWS:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    return {
        "average_score": average,
        "status_counts": status_counts,
        "rows": rows,
        "note": (
            "这是严格对标矩阵。当前 gushen 还不是完整 TradingAgents 复刻，"
            "而是一个可运行的 A股研究骨架，仍缺少多个高影响 Agent。"
        ),
    }


def write_alignment_doc(path: Path = Path("docs/TRADINGAGENTS_ALIGNMENT_MATRIX.md")) -> None:
    payload = build_alignment_payload()
    lines = [
        "# TradingAgents 对标矩阵",
        "",
        f"平均对标分：{payload['average_score']}/100",
        "",
        (
            "这张表刻意严格。只有在数据、执行流程和决策影响都接近原版时，"
            "组件才会得到高分；只有类名相似不算真正对标。"
        ),
        "",
        "| 原版组件 | 原版行为 | 当前 gushen 对应实现 | A股改造要求 | 状态 | 分数 | 下一步动作 |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in ALIGNMENT_ROWS:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.upstream_component,
                    row.upstream_behavior,
                    row.gushen_equivalent,
                    row.a_share_adaptation,
                    row.status,
                    str(row.score),
                    row.next_action,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("优先级：DataQualityAgent 闸门、SectorThemeAgent、FundFlowAgent、SentimentNarrativeAgent 原始流、MemoryReviewAgent，最后再做真正的工具调用图。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
