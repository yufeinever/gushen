from gushen.agents import StockContext, run_agents
from gushen.storage import LocalStore


def test_store_saves_universe_and_decisions(tmp_path) -> None:
    store = LocalStore(tmp_path / "gushen.sqlite")
    stock = StockContext(
        date="2026-05-21",
        code="600000.SH",
        name="Clean Sample",
        amount_rank=1,
        amount=1_000_000_000,
        pct_change=0.01,
        momentum_5d=0.02,
        volatility_20d=0.03,
    )
    state = run_agents(stock)

    store.initialize()
    store.save_universe([stock])
    store.save_decisions([state])

    with store.connect() as connection:
        universe_count = connection.execute("SELECT COUNT(*) FROM universe_snapshots").fetchone()[0]
        decision_count = connection.execute("SELECT COUNT(*) FROM agent_decisions").fetchone()[0]

    assert universe_count == 1
    assert decision_count == len(state.decisions)
