from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from gushen.agents import AgentDecision, CandidateState, StockContext


SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_snapshots (
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    amount_rank INTEGER NOT NULL,
    amount REAL NOT NULL,
    pct_change REAL NOT NULL,
    momentum_5d REAL NOT NULL,
    volatility_20d REAL NOT NULL,
    is_st INTEGER NOT NULL,
    is_suspended INTEGER NOT NULL,
    limit_status TEXT NOT NULL,
    event_tags TEXT NOT NULL,
    PRIMARY KEY (trade_date, code)
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    agent TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reasons TEXT NOT NULL,
    supporting_data TEXT NOT NULL,
    risks TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    invalid_condition TEXT NOT NULL,
    confidence_note TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, code, agent)
);
"""


class LocalStore:
    def __init__(self, path: str | Path = "data/local/gushen.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def save_universe(self, stocks: Iterable[StockContext]) -> None:
        rows = [
            (
                stock.date,
                stock.code,
                stock.name,
                stock.amount_rank,
                stock.amount,
                stock.pct_change,
                stock.momentum_5d,
                stock.volatility_20d,
                int(stock.is_st),
                int(stock.is_suspended),
                stock.limit_status,
                json.dumps(list(stock.event_tags), ensure_ascii=False),
            )
            for stock in stocks
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO universe_snapshots (
                    trade_date, code, name, amount_rank, amount, pct_change, momentum_5d,
                    volatility_20d, is_st, is_suspended, limit_status, event_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_decisions(self, states: Iterable[CandidateState]) -> None:
        rows: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        for state in states:
            rows.extend(self._decision_row(decision) for decision in state.decisions)

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO agent_decisions (
                    trade_date, code, agent, verdict, reasons, supporting_data, risks,
                    risk_level, invalid_condition, confidence_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    @staticmethod
    def _decision_row(decision: AgentDecision) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        return (
            decision.date,
            decision.code,
            decision.agent,
            decision.verdict,
            json.dumps(list(decision.reasons), ensure_ascii=False),
            json.dumps(list(decision.supporting_data), ensure_ascii=False),
            json.dumps(list(decision.risks), ensure_ascii=False),
            decision.risk_level,
            decision.invalid_condition,
            decision.confidence_note,
        )
