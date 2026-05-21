from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AgentAction = Literal["observe", "research", "paper_trade", "avoid"]
RiskLevel = Literal["low", "medium", "high"]
RiskStance = Literal["aggressive", "neutral", "conservative"]


class ResearchPlan(BaseModel):
    """Structured research conclusion inspired by TradingAgents' Research Manager."""

    recommendation: AgentAction = Field(
        description="Research recommendation constrained to non-real-money actions.",
    )
    thesis: str = Field(description="Main reason for the recommendation.")
    bull_points: list[str] = Field(default_factory=list)
    bear_points: list[str] = Field(default_factory=list)
    invalidation: str = Field(description="Condition that invalidates the research plan.")
    confidence_score: float = Field(ge=0.0, le=1.0)


class TraderPlan(BaseModel):
    """Structured paper-trading plan inspired by TradingAgents' Trader."""

    action: AgentAction
    entry_rule: str
    exit_rule: str
    holding_days: int = Field(ge=1, le=20)
    stop_loss_pct: float = Field(ge=0.0, le=0.3)
    max_position_pct: float = Field(ge=0.0, le=1.0)
    rationale: str


class RiskReview(BaseModel):
    """Single risk perspective in the risk debate."""

    stance: RiskStance
    verdict: AgentAction
    risk_level: RiskLevel
    key_points: list[str] = Field(default_factory=list)


class PortfolioDecision(BaseModel):
    """Final simulated portfolio decision."""

    final_action: AgentAction
    summary: str
    max_position_pct: float = Field(ge=0.0, le=1.0)
    risk_controls: list[str] = Field(default_factory=list)
    follow_up: str
