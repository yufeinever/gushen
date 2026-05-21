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
    )

    state = run_agents(stock)

    assert state.decisions[-1].verdict == "paper_trade"
