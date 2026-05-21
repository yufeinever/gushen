from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal["observe", "research", "paper_trade", "avoid"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class StockContext:
    date: str
    code: str
    name: str
    amount_rank: int
    amount: float
    pct_change: float
    momentum_5d: float
    volatility_20d: float
    is_st: bool = False
    is_suspended: bool = False
    limit_status: str = "none"
    event_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentDecision:
    date: str
    code: str
    agent: str
    verdict: Action
    reasons: tuple[str, ...] = ()
    supporting_data: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    risk_level: RiskLevel = "low"
    invalid_condition: str = ""
    confidence_note: str = ""


@dataclass
class CandidateState:
    stock: StockContext
    decisions: list[AgentDecision] = field(default_factory=list)

    def add(self, decision: AgentDecision) -> None:
        self.decisions.append(decision)

    @property
    def vetoed(self) -> bool:
        return any(decision.verdict == "avoid" and decision.risk_level == "high" for decision in self.decisions)


class BaseAgent:
    name = "BaseAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        raise NotImplementedError

    def _decision(
        self,
        stock: StockContext,
        verdict: Action,
        reasons: list[str],
        supporting_data: list[str] | None = None,
        risks: list[str] | None = None,
        risk_level: RiskLevel = "low",
        invalid_condition: str = "",
        confidence_note: str = "",
    ) -> AgentDecision:
        return AgentDecision(
            date=stock.date,
            code=stock.code,
            agent=self.name,
            verdict=verdict,
            reasons=tuple(reasons),
            supporting_data=tuple(supporting_data or []),
            risks=tuple(risks or []),
            risk_level=risk_level,
            invalid_condition=invalid_condition,
            confidence_note=confidence_note,
        )


class UniverseAgent(BaseAgent):
    name = "UniverseAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        risks: list[str] = []
        if stock.amount_rank > 100:
            risks.append("outside top 100 amount universe")
        if stock.is_st:
            risks.append("ST or special treatment status")
        if stock.is_suspended:
            risks.append("suspended security")

        if risks:
            return self._decision(
                stock,
                "avoid",
                ["failed universe eligibility checks"],
                risks=risks,
                risk_level="high",
                invalid_condition="stock returns to eligible high-liquidity universe",
            )

        return self._decision(
            stock,
            "research",
            ["eligible high-liquidity A-share candidate"],
            supporting_data=[f"amount_rank={stock.amount_rank}", f"amount={stock.amount:.2f}"],
        )


class TechnicalAnalystAgent(BaseAgent):
    name = "TechnicalAnalystAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        reasons: list[str] = []
        risks: list[str] = []

        if stock.momentum_5d > 0.03:
            reasons.append("positive 5-day momentum")
        elif stock.momentum_5d < -0.03:
            risks.append("negative 5-day momentum")

        if stock.pct_change > 0:
            reasons.append("closed higher on current session")
        else:
            risks.append("closed lower on current session")

        if stock.volatility_20d > 0.08:
            risks.append("20-day volatility is elevated")

        risk_level: RiskLevel = "medium" if risks else "low"
        verdict: Action = "research" if reasons else "observe"
        return self._decision(
            stock,
            verdict,
            reasons or ["no strong technical edge"],
            supporting_data=[
                f"pct_change={stock.pct_change:.4f}",
                f"momentum_5d={stock.momentum_5d:.4f}",
                f"volatility_20d={stock.volatility_20d:.4f}",
            ],
            risks=risks,
            risk_level=risk_level,
            invalid_condition="momentum reverses or volatility expands further",
        )


class EventAnalystAgent(BaseAgent):
    name = "EventAnalystAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        severe_tags = {"investigation", "litigation", "delisting", "impairment"}
        severe = sorted(set(stock.event_tags).intersection(severe_tags))

        if severe:
            return self._decision(
                stock,
                "avoid",
                ["material event risk detected"],
                risks=[f"event_tag={tag}" for tag in severe],
                risk_level="high",
                invalid_condition="material event risk is resolved and price behavior stabilizes",
            )

        if stock.event_tags:
            return self._decision(
                stock,
                "observe",
                ["non-severe event tags require monitoring"],
                supporting_data=[f"event_tag={tag}" for tag in stock.event_tags],
                risks=["event interpretation may be incomplete"],
                risk_level="medium",
            )

        return self._decision(stock, "research", ["no event risk tag in current sample"])


class BullResearcherAgent(BaseAgent):
    name = "BullResearcherAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        reasons = []
        if stock.amount_rank <= 30:
            reasons.append("top 30 liquidity concentrates market attention")
        if stock.momentum_5d > 0:
            reasons.append("short-term momentum is positive")
        if stock.pct_change > 0:
            reasons.append("current session confirms buying pressure")

        return self._decision(
            stock,
            "research" if len(reasons) >= 2 else "observe",
            reasons or ["bull case is weak in current sample"],
            confidence_note="bull case uses only deterministic sample features",
        )


class BearResearcherAgent(BaseAgent):
    name = "BearResearcherAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        risks = []
        if stock.volatility_20d > 0.08:
            risks.append("high volatility can amplify drawdown")
        if stock.limit_status in {"up_limit", "down_limit"}:
            risks.append(f"limit status is {stock.limit_status}")
        if stock.momentum_5d < 0:
            risks.append("negative short-term momentum")

        return self._decision(
            stock,
            "avoid" if len(risks) >= 3 else "observe",
            ["bear case evaluated"],
            risks=risks,
            risk_level="high" if len(risks) >= 3 else "medium" if risks else "low",
        )


class RiskManagerAgent(BaseAgent):
    name = "RiskManagerAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        high_risk = any(decision.risk_level == "high" for decision in state.decisions)
        avoid = any(decision.verdict == "avoid" for decision in state.decisions)

        if high_risk or avoid:
            return self._decision(
                stock,
                "avoid",
                ["risk veto triggered"],
                risks=[
                    f"{decision.agent}: {', '.join(decision.risks) or decision.verdict}"
                    for decision in state.decisions
                    if decision.risk_level == "high" or decision.verdict == "avoid"
                ],
                risk_level="high",
                invalid_condition="all high-risk veto reasons are cleared",
            )

        return self._decision(stock, "paper_trade", ["passes current deterministic risk checks"])


class PortfolioManagerAgent(BaseAgent):
    name = "PortfolioManagerAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        if state.vetoed:
            return self._decision(
                stock,
                "avoid",
                ["candidate rejected after risk review"],
                risks=["risk veto present"],
                risk_level="high",
            )

        research_votes = sum(decision.verdict == "research" for decision in state.decisions)
        paper_votes = sum(decision.verdict == "paper_trade" for decision in state.decisions)
        if paper_votes and research_votes >= 3:
            verdict: Action = "paper_trade"
            reasons = ["sufficient research support and risk checks passed"]
        elif research_votes >= 2:
            verdict = "research"
            reasons = ["needs more research before paper trading"]
        else:
            verdict = "observe"
            reasons = ["insufficient support for simulated action"]

        return self._decision(stock, verdict, reasons)


DEFAULT_AGENTS: tuple[BaseAgent, ...] = (
    UniverseAgent(),
    TechnicalAnalystAgent(),
    EventAnalystAgent(),
    BullResearcherAgent(),
    BearResearcherAgent(),
    RiskManagerAgent(),
    PortfolioManagerAgent(),
)


def run_agents(stock: StockContext, agents: tuple[BaseAgent, ...] = DEFAULT_AGENTS) -> CandidateState:
    state = CandidateState(stock=stock)
    for agent in agents:
        decision = agent.decide(state)
        state.add(decision)
    return state
