from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path
from typing import Any

from gushen.agents import StockContext


@dataclass(frozen=True)
class MarketFetchResult:
    trade_date: str
    stocks: list[StockContext]
    source: str


def load_sample_top_amount(path: str | Path = "data/samples/top_amount_sample.csv") -> MarketFetchResult:
    sample_path = Path(path)
    with sample_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"Sample file is empty: {sample_path}")

    stocks = [
        StockContext(
            date=row["date"],
            code=row["code"],
            name=row["name"],
            amount_rank=int(row["amount_rank"]),
            amount=float(row["amount"]),
            pct_change=float(row["pct_change"]),
            momentum_5d=float(row["momentum_5d"]),
            volatility_20d=float(row["volatility_20d"]),
            is_st=_to_bool(row["is_st"]),
            is_suspended=_to_bool(row["is_suspended"]),
            limit_status=row["limit_status"],
            event_tags=tuple(tag for tag in row["event_tags"].split("|") if tag),
        )
        for row in rows
    ]
    return MarketFetchResult(trade_date=stocks[0].date, stocks=stocks, source=str(sample_path))


def fetch_top_amount_stocks(limit: int = 100) -> MarketFetchResult:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "AKShare and pandas are required. Install with: pip install -e .[data]"
        ) from exc

    spot = _fetch_eastmoney_top_amount(limit=limit)
    if spot.empty:
        spot = ak.stock_zh_a_spot_em()
    if spot.empty:
        raise RuntimeError("AKShare returned an empty A-share spot table.")

    frame = _normalize_spot_frame(spot)
    frame = frame.sort_values("amount", ascending=False).head(limit).reset_index(drop=True)
    frame["amount_rank"] = frame.index + 1
    trade_date = datetime.now().strftime("%Y-%m-%d")

    stocks = [
        StockContext(
            date=trade_date,
            code=_to_ts_code(str(row["code"])),
            name=str(row["name"]),
            amount_rank=int(row["amount_rank"]),
            amount=float(row["amount"]),
            pct_change=float(row["pct_change"]) / 100.0,
            momentum_5d=0.0,
            volatility_20d=0.0,
            is_st=_is_st(str(row["name"])),
            is_suspended=False,
            limit_status="none",
        )
        for _, row in frame.iterrows()
    ]
    return MarketFetchResult(trade_date=trade_date, stocks=stocks, source="eastmoney.top_amount")


def _fetch_eastmoney_top_amount(limit: int):
    import pandas as pd
    import requests

    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    params: dict[str, Any] = {
        "pn": "1",
        "pz": str(max(limit, 30)),
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f6",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f12,f14,f3,f6",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("diff", [])
    if not rows:
        return pd.DataFrame(columns=["代码", "名称", "涨跌幅", "成交额"])

    return pd.DataFrame(
        [
            {
                "代码": row.get("f12"),
                "名称": row.get("f14"),
                "涨跌幅": row.get("f3"),
                "成交额": row.get("f6"),
            }
            for row in rows
        ]
    )


def _normalize_spot_frame(frame):
    import pandas as pd

    column_map = {
        "代码": "code",
        "名称": "name",
        "成交额": "amount",
        "涨跌幅": "pct_change",
    }
    missing = [column for column in column_map if column not in frame.columns]
    if missing:
        raise RuntimeError(f"AKShare spot table missing expected columns: {missing}")

    normalized = frame.rename(columns=column_map)[list(column_map.values())].copy()
    normalized["amount"] = pd.to_numeric(normalized["amount"], errors="coerce").fillna(0)
    normalized["pct_change"] = pd.to_numeric(normalized["pct_change"], errors="coerce").fillna(0)
    normalized = normalized[normalized["amount"] > 0]
    normalized = normalized[~normalized["name"].map(_is_st)]
    return normalized


def _to_ts_code(code: str) -> str:
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return code


def _is_st(name: str) -> bool:
    upper = name.upper()
    return "ST" in upper or "退" in name


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}
