from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import csv
from pathlib import Path
from typing import Any

from gushen.agents import StockContext
from gushen.domestic_network import direct_requests_get, domestic_data_no_proxy


@dataclass(frozen=True)
class MarketFetchResult:
    trade_date: str
    stocks: list[StockContext]
    source: str


@dataclass(frozen=True)
class DailyBar:
    trade_date: str
    code: str
    name: str
    open: float
    close: float
    high: float
    low: float
    volume: float
    amount: float
    amplitude: float
    pct_change: float
    turnover: float


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

    with domestic_data_no_proxy():
        spot = _fetch_eastmoney_top_amount(limit=limit)
    if spot.empty:
        with domestic_data_no_proxy():
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
    response = direct_requests_get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("diff", [])
    if not rows:
        return pd.DataFrame(columns=[_c("code"), _c("name"), _c("pct_change"), _c("amount")])

    return pd.DataFrame(
        [
            {
                _c("code"): row.get("f12"),
                _c("name"): row.get("f14"),
                _c("pct_change"): row.get("f3"),
                _c("amount"): row.get("f6"),
            }
            for row in rows
        ]
    )


def _normalize_spot_frame(frame):
    import pandas as pd

    column_map = {
        _c("code"): "code",
        _c("name"): "name",
        _c("amount"): "amount",
        _c("pct_change"): "pct_change",
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


def fetch_a_share_code_names() -> list[tuple[str, str]]:
    import akshare as ak

    frame = ak.stock_info_a_code_name()
    return [(str(row["code"]).zfill(6), str(row["name"])) for _, row in frame.iterrows()]


def fetch_daily_bar(code: str, name: str, trade_date: str, timeout: float = 12) -> DailyBar | None:
    try:
        return _fetch_daily_bar_eastmoney(code, name, trade_date, timeout)
    except Exception:
        return _fetch_daily_bar_tencent(code, name, trade_date, timeout)


def _fetch_daily_bar_eastmoney(
    code: str,
    name: str,
    trade_date: str,
    timeout: float = 12,
) -> DailyBar | None:
    import requests

    date_arg = trade_date.replace("-", "")
    market_code = 1 if code.startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": f"{market_code}.{code}",
        "beg": date_arg,
        "end": date_arg,
    }
    response = direct_requests_get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    klines = payload.get("data", {}).get("klines") if payload.get("data") else None
    if not klines:
        return None

    values = klines[0].split(",")
    return DailyBar(
        trade_date=values[0],
        code=_to_ts_code(code),
        name=name,
        open=float(values[1]),
        close=float(values[2]),
        high=float(values[3]),
        low=float(values[4]),
        volume=float(values[5]),
        amount=float(values[6]),
        amplitude=float(values[7]) / 100.0,
        pct_change=float(values[8]) / 100.0,
        turnover=float(values[10]) / 100.0,
    )


def _fetch_daily_bar_tencent(
    code: str,
    name: str,
    trade_date: str,
    timeout: float = 12,
) -> DailyBar | None:
    start_date = (date.fromisoformat(trade_date) - timedelta(days=10)).strftime("%Y%m%d")
    rows = _fetch_daily_bars_tencent(
        _to_ts_code(code),
        name,
        start_date,
        trade_date.replace("-", ""),
        timeout,
    )
    exact = [row for row in rows if row.trade_date == trade_date]
    return exact[-1] if exact else None


def fetch_daily_bars(
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    timeout: float = 20,
    adjust: str = "qfq",
) -> list[DailyBar]:
    try:
        rows = _fetch_daily_bars_eastmoney(code, name, start_date, end_date, timeout, adjust)
        if rows:
            return rows
    except Exception:
        pass
    return _fetch_daily_bars_tencent(code, name, start_date, end_date, timeout, adjust)


def _fetch_daily_bars_eastmoney(
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    timeout: float = 20,
    adjust: str = "qfq",
) -> list[DailyBar]:
    import requests

    start_arg = start_date.replace("-", "")
    end_arg = end_date.replace("-", "")
    raw_code = code.split(".")[0]
    market_code = 1 if raw_code.startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": _eastmoney_adjust_flag(adjust),
        "secid": f"{market_code}.{raw_code}",
        "beg": start_arg,
        "end": end_arg,
    }
    response = direct_requests_get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    klines = payload.get("data", {}).get("klines") if payload.get("data") else None
    if not klines:
        return []

    bars = []
    for kline in klines:
        values = kline.split(",")
        bars.append(
            DailyBar(
                trade_date=values[0],
                code=_to_ts_code(raw_code),
                name=name,
                open=float(values[1]),
                close=float(values[2]),
                high=float(values[3]),
                low=float(values[4]),
                volume=float(values[5]),
                amount=float(values[6]),
                amplitude=float(values[7]) / 100.0,
                pct_change=float(values[8]) / 100.0,
                turnover=float(values[10]) / 100.0,
            )
        )
    return bars


def _fetch_daily_bars_tencent(
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    timeout: float = 20,
    adjust: str = "qfq",
) -> list[DailyBar]:
    import akshare as ak

    raw_code = code.split(".")[0]
    symbol = _to_tencent_symbol(raw_code)
    start_arg = start_date.replace("-", "")
    end_arg = end_date.replace("-", "")
    with domestic_data_no_proxy():
        frame = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            start_date=start_arg,
            end_date=end_arg,
            adjust=_tencent_adjust_flag(adjust),
            timeout=timeout,
        )
    if frame.empty:
        return []

    rows = []
    previous_close: float | None = None
    for _, item in frame.sort_values("date").iterrows():
        trade_date = str(item["date"])
        close = _safe_float(item["close"])
        open_price = _safe_float(item["open"])
        high = _safe_float(item["high"])
        low = _safe_float(item["low"])
        volume_hands = _safe_float(item["amount"])
        pct_change = (close / previous_close - 1.0) if previous_close else 0.0
        amount = volume_hands * 100.0 * close
        amplitude = ((high - low) / previous_close) if previous_close else 0.0
        rows.append(
            DailyBar(
                trade_date=trade_date,
                code=_to_ts_code(raw_code),
                name=name,
                open=open_price,
                close=close,
                high=high,
                low=low,
                volume=volume_hands * 100.0,
                amount=amount,
                amplitude=amplitude,
                pct_change=pct_change,
                turnover=0.0,
            )
        )
        previous_close = close
    return rows


def _to_tencent_symbol(code: str) -> str:
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    return f"{prefix}{code}"


def _eastmoney_adjust_flag(adjust: str) -> str:
    values = {"none": "0", "qfq": "1", "hfq": "2"}
    return values.get(adjust, "1")


def _tencent_adjust_flag(adjust: str) -> str:
    values = {"none": "", "qfq": "qfq", "hfq": "hfq"}
    return values.get(adjust, "qfq")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _c(name: str) -> str:
    columns = {
        "code": "\u4ee3\u7801",
        "name": "\u540d\u79f0",
        "amount": "\u6210\u4ea4\u989d",
        "pct_change": "\u6da8\u8dcc\u5e45",
    }
    return columns[name]
