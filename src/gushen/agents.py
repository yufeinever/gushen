from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from gushen.agent_schemas import (
    PortfolioDecision,
    ResearchPlan,
    RiskReview,
    SentimentNarrative,
    TraderPlan,
)

Action = Literal["observe", "research", "paper_trade", "avoid"]
RiskLevel = Literal["low", "medium", "high"]
QualityStatus = Literal["unknown", "paper_trade_ready", "research_only", "blocked"]


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
    data_quality_status: QualityStatus = "unknown"
    data_quality_score: float = 0.0
    data_quality_gaps: tuple[str, ...] = ()


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
    artifacts: dict[str, object] = field(default_factory=dict)

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


class DataQualityAgent(BaseAgent):
    name = "DataQualityAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        gaps = list(stock.data_quality_gaps)
        if stock.data_quality_status == "blocked":
            return self._decision(
                stock,
                "avoid",
                ["data quality gate blocks analysis"],
                supporting_data=[f"data_quality_score={stock.data_quality_score:.2f}"],
                risks=gaps or ["core data is insufficient"],
                risk_level="high",
                invalid_condition="core market, tradability and backtest data pass the quality gate",
            )
        if stock.data_quality_status in {"research_only", "unknown"}:
            return self._decision(
                stock,
                "research",
                ["data quality gate allows research only"],
                supporting_data=[f"data_quality_score={stock.data_quality_score:.2f}"],
                risks=gaps[:5] or ["dataset has not been marked paper-trade ready"],
                risk_level="medium",
                invalid_condition="missing A-share context feeds are filled and validated",
            )
        return self._decision(
            stock,
            "research",
            ["data quality gate passes for simulated research"],
            supporting_data=[f"data_quality_score={stock.data_quality_score:.2f}"],
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


class SentimentNarrativeAgent(BaseAgent):
    name = "SentimentNarrativeAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        text = str(state.artifacts.get("event_summary", ""))
        narratives = _extract_narratives(text)
        unsupported = []
        verification = []
        if not text:
            verification.append("missing event/news/forum narrative text")
        if not any(keyword in text for keyword in ["公告", "年报", "季报", "龙虎榜", "交易所"]):
            unsupported.append("narrative lacks official filing or exchange-backed evidence")
        if any(keyword in text for keyword in ["龙虎榜", "涨停", "主力", "融资"]):
            crowding: RiskLevel = "high"
        elif narratives:
            crowding = "medium"
        else:
            crowding = "low"

        narrative = SentimentNarrative(
            dominant_narratives=narratives[:3],
            counter_narratives=[],
            evidence_backed_claims=[
                item for item in narratives if any(key in item for key in ["公告", "年报", "季报", "龙虎榜"])
            ],
            unsupported_claims=unsupported,
            crowding_risk=crowding,
            verification_needed=verification or ["verify narrative against announcement/news source"],
            quality_score=0.65 if narratives and not unsupported else 0.35 if text else 0.1,
        )
        state.artifacts["sentiment_narrative"] = narrative
        risks = list(narrative.unsupported_claims)
        if narrative.crowding_risk == "high":
            risks.append("narrative crowding risk is high")
        return self._decision(
            stock,
            "research" if narrative.quality_score >= 0.6 else "observe",
            narrative.dominant_narratives or ["no usable narrative text"],
            supporting_data=[f"quality_score={narrative.quality_score:.2f}"],
            risks=risks,
            risk_level="high" if narrative.crowding_risk == "high" else "medium" if risks else "low",
            invalid_condition="narrative cannot be verified by reliable source",
        )


class BullResearcherAgent(BaseAgent):
    name = "BullResearcherAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        reasons = []
        if stock.amount_rank <= 100:
            reasons.append("candidate is inside the top 100 amount universe")
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


class ResearchManagerAgent(BaseAgent):
    name = "ResearchManagerAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        high_risk = [decision for decision in state.decisions if decision.risk_level == "high"]
        research_only = any(decision.agent == "DataQualityAgent" and decision.risk_level == "medium" for decision in state.decisions)
        bull_points = [
            reason
            for decision in state.decisions
            if decision.agent in {"BullResearcherAgent", "SentimentNarrativeAgent"}
            for reason in decision.reasons
        ]
        bear_points = [
            risk
            for decision in state.decisions
            if decision.agent in {"BearResearcherAgent", "TechnicalAnalystAgent", "EventAnalystAgent"}
            for risk in decision.risks
        ]
        research_votes = sum(decision.verdict == "research" for decision in state.decisions)

        if high_risk:
            recommendation: Action = "avoid"
            confidence = 0.85
            thesis = "High-risk veto appeared before research synthesis."
        elif research_only:
            recommendation = "research"
            confidence = 0.52
            thesis = "Data quality gate allows research only, so no simulated trade is approved."
        elif research_votes >= 4 and not bear_points:
            recommendation = "paper_trade"
            confidence = 0.72
            thesis = "Multiple research agents support a simulated observation trade."
        elif research_votes >= 2:
            recommendation = "research"
            confidence = 0.58
            thesis = "Evidence is constructive but still needs more confirmation."
        else:
            recommendation = "observe"
            confidence = 0.45
            thesis = "Signal support is not strong enough for simulated trading."

        plan = ResearchPlan(
            recommendation=recommendation,
            thesis=thesis,
            bull_points=bull_points,
            bear_points=bear_points,
            invalidation="Breaks below recent support, event risk appears, or liquidity leaves Top100.",
            confidence_score=confidence,
        )
        state.artifacts["research_plan"] = plan
        return self._decision(
            stock,
            recommendation,
            [plan.thesis],
            supporting_data=[f"confidence_score={plan.confidence_score:.2f}"],
            risks=plan.bear_points,
            risk_level="high" if recommendation == "avoid" else "medium" if plan.bear_points else "low",
            invalid_condition=plan.invalidation,
        )


class TraderAgent(BaseAgent):
    name = "TraderAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        research_plan = state.artifacts.get("research_plan")
        if not isinstance(research_plan, ResearchPlan):
            return self._decision(
                stock,
                "avoid",
                ["missing research plan"],
                risks=["ResearchManagerAgent did not produce a plan"],
                risk_level="high",
            )

        if research_plan.recommendation == "paper_trade":
            action: Action = "paper_trade"
            max_position = 0.05
            stop_loss = 0.04 if stock.volatility_20d <= 0.04 else 0.06
            rationale = "Translate the research plan into a small simulated T+1 trade."
        elif research_plan.recommendation == "research":
            action = "research"
            max_position = 0.0
            stop_loss = 0.0
            rationale = "Keep it in research until the next daily signal confirms."
        else:
            action = research_plan.recommendation
            max_position = 0.0
            stop_loss = 0.0
            rationale = "No simulated entry because research plan is not constructive."

        plan = TraderPlan(
            action=action,
            entry_rule="Next trading day only; use open/VWAP simulation, never same-day close.",
            exit_rule="Test 1/3/5 trading-day exits and compare against invalidation.",
            holding_days=3,
            stop_loss_pct=stop_loss,
            max_position_pct=max_position,
            rationale=rationale,
        )
        state.artifacts["trader_plan"] = plan
        return self._decision(
            stock,
            action,
            [plan.rationale],
            supporting_data=[
                f"holding_days={plan.holding_days}",
                f"max_position_pct={plan.max_position_pct:.2%}",
                f"stop_loss_pct={plan.stop_loss_pct:.2%}",
            ],
            risks=[] if action == "paper_trade" else ["no executable simulated setup"],
            risk_level="low" if action == "paper_trade" else "medium" if action == "research" else "high",
        )


class AggressiveRiskAgent(BaseAgent):
    name = "AggressiveRiskAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        trader_plan = state.artifacts.get("trader_plan")
        supports_trade = isinstance(trader_plan, TraderPlan) and trader_plan.action == "paper_trade"
        review = RiskReview(
            stance="aggressive",
            verdict="paper_trade" if supports_trade else "observe",
            risk_level="medium" if stock.volatility_20d > 0.08 else "low",
            key_points=["liquidity can support a small simulated trade"]
            if supports_trade
            else ["no clear simulated trade to champion"],
        )
        state.artifacts["risk_aggressive"] = review
        return self._decision(
            stock,
            review.verdict,
            review.key_points,
            risk_level=review.risk_level,
        )


class ConservativeRiskAgent(BaseAgent):
    name = "ConservativeRiskAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        risks = []
        if stock.volatility_20d > 0.08:
            risks.append("volatility too high for first-stage simulation")
        if stock.limit_status in {"up_limit", "down_limit"}:
            risks.append("limit status may block execution")
        if any(decision.risk_level == "high" for decision in state.decisions):
            risks.append("prior high-risk flag exists")

        review = RiskReview(
            stance="conservative",
            verdict="avoid" if risks else "paper_trade",
            risk_level="high" if risks else "low",
            key_points=risks or ["no conservative veto"],
        )
        state.artifacts["risk_conservative"] = review
        return self._decision(
            stock,
            review.verdict,
            review.key_points,
            risks=risks,
            risk_level=review.risk_level,
        )


class NeutralRiskAgent(BaseAgent):
    name = "NeutralRiskAgent"

    def decide(self, state: CandidateState) -> AgentDecision:
        stock = state.stock
        aggressive = state.artifacts.get("risk_aggressive")
        conservative = state.artifacts.get("risk_conservative")
        conservative_veto = isinstance(conservative, RiskReview) and conservative.verdict == "avoid"
        aggressive_support = isinstance(aggressive, RiskReview) and aggressive.verdict == "paper_trade"
        verdict: Action = "avoid" if conservative_veto else "paper_trade" if aggressive_support else "observe"
        review = RiskReview(
            stance="neutral",
            verdict=verdict,
            risk_level="high" if conservative_veto else "low" if aggressive_support else "medium",
            key_points=[
                "conservative veto dominates"
                if conservative_veto
                else "aggressive support accepted for simulation"
                if aggressive_support
                else "risk debate is inconclusive"
            ],
        )
        state.artifacts["risk_neutral"] = review
        return self._decision(stock, review.verdict, review.key_points, risk_level=review.risk_level)


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

        trader_plan = state.artifacts.get("trader_plan")
        neutral_review = state.artifacts.get("risk_neutral")
        research_votes = sum(decision.verdict == "research" for decision in state.decisions)
        if (
            isinstance(trader_plan, TraderPlan)
            and isinstance(neutral_review, RiskReview)
            and trader_plan.action == "paper_trade"
            and neutral_review.verdict == "paper_trade"
        ):
            verdict: Action = "paper_trade"
            reasons = ["trader plan and neutral risk review approve simulation"]
        elif research_votes >= 2:
            verdict = "research"
            reasons = ["needs more research before paper trading"]
        else:
            verdict = "observe"
            reasons = ["insufficient support for simulated action"]

        decision = PortfolioDecision(
            final_action=verdict,
            summary=reasons[0],
            max_position_pct=trader_plan.max_position_pct if isinstance(trader_plan, TraderPlan) else 0.0,
            risk_controls=[
                "T+1 execution only",
                "no real order placement",
                "reject if outside Top100 amount universe",
            ],
            follow_up="Record next-session simulated entry and 1/3/5-day outcomes.",
        )
        state.artifacts["portfolio_decision"] = decision
        return self._decision(
            stock,
            verdict,
            reasons,
            supporting_data=[f"max_position_pct={decision.max_position_pct:.2%}"],
            risks=[] if verdict == "paper_trade" else ["not approved for simulation"],
            risk_level="low" if verdict == "paper_trade" else "medium",
        )


LEGACY_AGENTS: tuple[BaseAgent, ...] = (
    UniverseAgent(),
    TechnicalAnalystAgent(),
    EventAnalystAgent(),
    SentimentNarrativeAgent(),
    BullResearcherAgent(),
    BearResearcherAgent(),
    RiskManagerAgent(),
    PortfolioManagerAgent(),
)


DEFAULT_AGENTS: tuple[BaseAgent, ...] = (
    DataQualityAgent(),
    UniverseAgent(),
    TechnicalAnalystAgent(),
    EventAnalystAgent(),
    SentimentNarrativeAgent(),
    BullResearcherAgent(),
    BearResearcherAgent(),
    ResearchManagerAgent(),
    TraderAgent(),
    AggressiveRiskAgent(),
    ConservativeRiskAgent(),
    NeutralRiskAgent(),
    RiskManagerAgent(),
    PortfolioManagerAgent(),
)


def run_agents(stock: StockContext, agents: tuple[BaseAgent, ...] = DEFAULT_AGENTS) -> CandidateState:
    state = CandidateState(stock=stock)
    for agent in agents:
        decision = agent.decide(state)
        state.add(decision)
    return state


def _extract_narratives(text: str) -> list[str]:
    if not text:
        return []
    parts = [part.strip() for part in text.replace("\n", " | ").split("|")]
    return [part for part in parts if part][:5]
