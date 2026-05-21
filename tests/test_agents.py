from gushen.agent_schemas import PortfolioDecision, ResearchPlan, RiskReview, TraderPlan
from gushen.agents import StockContext, run_agents


def test_high_risk_event_is_vetoed() -> None:
    stock = StockContext(
        date="2026-05-21",
        code="300000.SZ",
        name="Risk Sample",
        amount_rank=10,
        amount=10_000_000_000,
        pct_change=0.03,
        momentum_5d=0.05,
        volatility_20d=0.04,
        event_tags=("investigation",),
    )

    state = run_agents(stock)

    assert state.decisions[-1].verdict == "avoid"
    assert state.vetoed is True
    assert isinstance(state.artifacts["research_plan"], ResearchPlan)
    assert state.artifacts["research_plan"].recommendation == "avoid"


def test_clean_liquid_candidate_can_reach_paper_trade() -> None:
    stock = StockContext(
        date="2026-05-21",
        code="600000.SH",
        name="Clean Sample",
        amount_rank=8,
        amount=12_000_000_000,
        pct_change=0.02,
        momentum_5d=0.06,
        volatility_20d=0.03,
        data_quality_status="paper_trade_ready",
        data_quality_score=90,
    )

    state = run_agents(stock)

    assert state.decisions[-1].verdict == "paper_trade"
    assert isinstance(state.artifacts["research_plan"], ResearchPlan)
    assert isinstance(state.artifacts["trader_plan"], TraderPlan)
    assert isinstance(state.artifacts["risk_neutral"], RiskReview)
    assert isinstance(state.artifacts["portfolio_decision"], PortfolioDecision)
    assert state.artifacts["portfolio_decision"].final_action == "paper_trade"


def test_research_only_quality_gate_blocks_paper_trade() -> None:
    stock = StockContext(
        date="2026-05-21",
        code="000001.SZ",
        name="Research Only",
        amount_rank=5,
        amount=15_000_000_000,
        pct_change=0.03,
        momentum_5d=0.06,
        volatility_20d=0.03,
        data_quality_status="research_only",
        data_quality_score=62,
        data_quality_gaps=("fund flow missing", "sector theme missing"),
    )

    state = run_agents(stock)

    assert state.decisions[0].agent == "DataQualityAgent"
    assert state.artifacts["research_plan"].recommendation == "research"
    assert state.decisions[-1].verdict == "research"
