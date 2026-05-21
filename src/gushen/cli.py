from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from gushen.agents import StockContext, run_agents


SAMPLE_STOCKS = [
    StockContext(
        date="2026-05-21",
        code="600000.SH",
        name="Sample Bank",
        amount_rank=12,
        amount=8_500_000_000,
        pct_change=0.018,
        momentum_5d=0.052,
        volatility_20d=0.032,
    ),
    StockContext(
        date="2026-05-21",
        code="000001.SZ",
        name="Sample Finance",
        amount_rank=27,
        amount=6_200_000_000,
        pct_change=-0.011,
        momentum_5d=-0.036,
        volatility_20d=0.041,
        event_tags=("pledge",),
    ),
    StockContext(
        date="2026-05-21",
        code="300000.SZ",
        name="Sample Tech",
        amount_rank=5,
        amount=13_400_000_000,
        pct_change=0.044,
        momentum_5d=0.087,
        volatility_20d=0.094,
        event_tags=("investigation",),
    ),
]


def _final_action(state) -> str:
    return state.decisions[-1].verdict


def main() -> None:
    console = Console()
    states = [run_agents(stock) for stock in SAMPLE_STOCKS]

    table = Table(title="GushenAgents demo")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Rank", justify="right")
    table.add_column("Final Action")
    table.add_column("Risk")

    for state in states:
        final = state.decisions[-1]
        table.add_row(
            state.stock.code,
            state.stock.name,
            str(state.stock.amount_rank),
            final.verdict,
            final.risk_level,
        )

    console.print(table)
    console.print()
    console.print_json(
        json.dumps(
            [
                {
                    "stock": {
                        "date": state.stock.date,
                        "code": state.stock.code,
                        "name": state.stock.name,
                        "amount_rank": state.stock.amount_rank,
                    },
                    "final_action": _final_action(state),
                    "decisions": [decision.__dict__ for decision in state.decisions],
                }
                for state in states
            ],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
