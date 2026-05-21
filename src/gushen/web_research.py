from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebResearchRequest:
    query: str
    purpose: str
    preferred_sources: tuple[str, ...] = ()


RELIABLE_A_SHARE_SOURCES = (
    "cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
    "bse.cn",
    "csrc.gov.cn",
    "eastmoney.com",
    "cs.com.cn",
    "stcn.com",
)


def build_event_search_request(code: str, name: str, trade_date: str) -> WebResearchRequest:
    return WebResearchRequest(
        query=f"{code} {name} 公告 新闻 风险 {trade_date}",
        purpose="补充本地事件数据缺口，优先核验公告、交易所披露、监管和可靠财经新闻。",
        preferred_sources=RELIABLE_A_SHARE_SOURCES,
    )
