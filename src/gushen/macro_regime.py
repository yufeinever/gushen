from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from gushen.domestic_network import domestic_data_no_proxy


DEFAULT_TRADE_DATE = "2026-05-20"


def _zh(key: str) -> str:
    values = {
        "date": "\u65e5\u671f",
        "us10y": "\u7f8e\u56fd\u56fd\u503a\u6536\u76ca\u738710\u5e74",
        "us30y": "\u7f8e\u56fd\u56fd\u503a\u6536\u76ca\u738730\u5e74",
        "cn10y": "\u4e2d\u56fd\u56fd\u503a\u6536\u76ca\u738710\u5e74",
        "cny_latest": "\u6700\u65b0\u4ef7",
        "lpr_date": "TRADE_DATE",
        "lpr1y": "LPR1Y",
        "lpr5y": "LPR5Y",
        "shibor_date": "\u65e5\u671f",
        "shibor_on": "O/N-\u5b9a\u4ef7",
        "shibor_3m": "3M-\u5b9a\u4ef7",
        "pmi_month": "\u6708\u4efd",
        "pmi_mfg": "\u5236\u9020\u4e1a-\u6307\u6570",
    }
    return values[key]


@dataclass(frozen=True)
class MacroRegime:
    trade_date: str
    status: str
    score_adjustment: float
    us_10y: float | None
    us_30y: float | None
    cn_10y: float | None
    usdcnh: float | None
    lpr_1y: float | None
    lpr_5y: float | None
    shibor_on: float | None
    shibor_3m: float | None
    china_pmi: float | None
    qvix_300etf: float | None
    risks: list[str]
    supports: list[str]
    sources: list[str]


def build_macro_regime(trade_date: str = DEFAULT_TRADE_DATE) -> MacroRegime:
    regime = fetch_macro_regime(trade_date)
    write_macro_regime(regime)
    print_macro_regime(Console(), regime)
    return regime


def load_or_build_macro_regime(trade_date: str = DEFAULT_TRADE_DATE) -> MacroRegime:
    path = Path(f"reports/generated/macro_regime_{trade_date}.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return MacroRegime(**data)
    return build_macro_regime(trade_date)


def fetch_macro_regime(trade_date: str = DEFAULT_TRADE_DATE) -> MacroRegime:
    values: dict[str, Any] = {}
    sources = []
    with domestic_data_no_proxy():
        values.update(_fetch_bond_rates(trade_date))
        sources.append("AKShare bond_zh_us_rate")
        values.update(_fetch_usdcnh())
        sources.append("AKShare forex_hist_em(USDCNH)")
        values.update(_fetch_lpr())
        sources.append("AKShare macro_china_lpr")
        values.update(_fetch_shibor())
        sources.append("AKShare macro_china_shibor_all")
        values.update(_fetch_china_pmi())
        sources.append("AKShare macro_china_pmi")
        values.update(_fetch_qvix())
        sources.append("AKShare index_option_300etf_qvix")

    risks: list[str] = []
    supports: list[str] = []
    score_adjustment = 0.0
    us_30y = values.get("us_30y")
    us_10y = values.get("us_10y")
    usdcnh = values.get("usdcnh")
    qvix = values.get("qvix_300etf")
    pmi = values.get("china_pmi")
    shibor_on = values.get("shibor_on")

    if us_30y is not None and us_30y >= 5.0:
        risks.append(f"\u7f8e\u503a30\u5e74\u6536\u76ca\u7387\u9ad8\u4f4d {us_30y:.2f}%\uff0c\u538b\u5236\u9ad8\u4f30\u503c\u6210\u957f\u80a1")
        score_adjustment -= 8
    if us_10y is not None and us_10y >= 4.5:
        risks.append(f"\u7f8e\u503a10\u5e74\u6536\u76ca\u7387 {us_10y:.2f}%\uff0c\u5168\u7403\u98ce\u9669\u504f\u597d\u627f\u538b")
        score_adjustment -= 5
    if usdcnh is not None and usdcnh >= 7.0:
        risks.append(f"USDCNH {usdcnh:.4f}\uff0c\u4eba\u6c11\u5e01\u8d2c\u503c\u538b\u529b\u504f\u9ad8")
        score_adjustment -= 4
    elif usdcnh is not None:
        supports.append(f"USDCNH {usdcnh:.4f}\uff0c\u6c47\u7387\u538b\u529b\u6682\u672a\u8d8a\u8fc7 7.0")
    if qvix is not None and qvix >= 25:
        risks.append(f"300ETF QVIX {qvix:.2f}\uff0cA\u80a1\u6ce2\u52a8\u98ce\u9669\u504f\u9ad8")
        score_adjustment -= 4
    elif qvix is not None and qvix <= 20:
        supports.append(f"300ETF QVIX {qvix:.2f}\uff0c\u671f\u6743\u9690\u542b\u6ce2\u52a8\u4e0d\u9ad8")
        score_adjustment += 2
    if pmi is not None and pmi < 50:
        risks.append(f"\u5236\u9020\u4e1a PMI {pmi:.1f}\uff0c\u7ecf\u6d4e\u666f\u6c14\u4ecd\u504f\u5f31")
        score_adjustment -= 3
    elif pmi is not None:
        supports.append(f"\u5236\u9020\u4e1a PMI {pmi:.1f}\uff0c\u666f\u6c14\u4f4d\u4e8e\u6269\u5f20\u7ebf\u4e0a")
        score_adjustment += 2
    if shibor_on is not None and shibor_on <= 1.5:
        supports.append(f"\u9694\u591c SHIBOR {shibor_on:.3f}%\uff0c\u56fd\u5185\u77ed\u7aef\u6d41\u52a8\u6027\u5c1a\u53ef")
        score_adjustment += 2

    if score_adjustment <= -10:
        status = "high_risk"
    elif score_adjustment < 0:
        status = "neutral_tight"
    else:
        status = "supportive"

    return MacroRegime(
        trade_date=trade_date,
        status=status,
        score_adjustment=round(score_adjustment, 2),
        us_10y=us_10y,
        us_30y=us_30y,
        cn_10y=values.get("cn_10y"),
        usdcnh=usdcnh,
        lpr_1y=values.get("lpr_1y"),
        lpr_5y=values.get("lpr_5y"),
        shibor_on=shibor_on,
        shibor_3m=values.get("shibor_3m"),
        china_pmi=pmi,
        qvix_300etf=qvix,
        risks=risks,
        supports=supports,
        sources=sources,
    )


def write_macro_regime(regime: MacroRegime) -> None:
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"macro_regime_{regime.trade_date}.json").write_text(
        json.dumps(asdict(regime), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / f"macro_regime_{regime.trade_date}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(MacroRegime.__dataclass_fields__))
        writer.writeheader()
        writer.writerow(asdict(regime))


def print_macro_regime(console: Console, regime: MacroRegime) -> None:
    table = Table(title=f"{regime.trade_date} MacroRegimeAgent")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("status", regime.status)
    table.add_row("score_adjustment", f"{regime.score_adjustment:.2f}")
    table.add_row("US 10Y / 30Y", f"{regime.us_10y} / {regime.us_30y}")
    table.add_row("USDCNH", str(regime.usdcnh))
    table.add_row("LPR 1Y / 5Y", f"{regime.lpr_1y} / {regime.lpr_5y}")
    table.add_row("SHIBOR O/N / 3M", f"{regime.shibor_on} / {regime.shibor_3m}")
    table.add_row("China PMI", str(regime.china_pmi))
    table.add_row("300ETF QVIX", str(regime.qvix_300etf))
    console.print(table)
    console.print("Risks: " + "; ".join(regime.risks or ["none"]))
    console.print("Supports: " + "; ".join(regime.supports or ["none"]))


def _fetch_bond_rates(trade_date: str) -> dict[str, float | None]:
    import akshare as ak

    frame = ak.bond_zh_us_rate(start_date=trade_date.replace("-", ""))
    if frame is None or frame.empty:
        return {}
    row = frame.tail(1).iloc[0]
    return {
        "us_10y": _float_or_none(row.get(_zh("us10y"))),
        "us_30y": _float_or_none(row.get(_zh("us30y"))),
        "cn_10y": _float_or_none(row.get(_zh("cn10y"))),
    }


def _fetch_usdcnh() -> dict[str, float | None]:
    import akshare as ak

    frame = ak.forex_hist_em(symbol="USDCNH")
    if frame is None or frame.empty:
        return {}
    row = frame.tail(1).iloc[0]
    return {"usdcnh": _float_or_none(row.get(_zh("cny_latest")))}


def _fetch_lpr() -> dict[str, float | None]:
    import akshare as ak

    frame = ak.macro_china_lpr()
    if frame is None or frame.empty:
        return {}
    row = frame.tail(1).iloc[0]
    return {
        "lpr_1y": _float_or_none(row.get(_zh("lpr1y"))),
        "lpr_5y": _float_or_none(row.get(_zh("lpr5y"))),
    }


def _fetch_shibor() -> dict[str, float | None]:
    import akshare as ak

    frame = ak.macro_china_shibor_all()
    if frame is None or frame.empty:
        return {}
    row = frame.tail(1).iloc[0]
    return {
        "shibor_on": _float_or_none(row.get(_zh("shibor_on"))),
        "shibor_3m": _float_or_none(row.get(_zh("shibor_3m"))),
    }


def _fetch_china_pmi() -> dict[str, float | None]:
    import akshare as ak
    import pandas as pd

    frame = ak.macro_china_pmi()
    if frame is None or frame.empty:
        return {}
    work = frame.copy()
    work["parsed"] = pd.to_datetime(
        work[_zh("pmi_month")].astype(str).str.replace("\u5e74", "-").str.replace("\u6708\u4efd", "-01"),
        errors="coerce",
    )
    work = work.dropna(subset=["parsed"]).sort_values("parsed")
    if work.empty:
        return {}
    row = work.tail(1).iloc[0]
    return {"china_pmi": _float_or_none(row.get(_zh("pmi_mfg")))}


def _fetch_qvix() -> dict[str, float | None]:
    import akshare as ak

    frame = ak.index_option_300etf_qvix()
    if frame is None or frame.empty:
        return {}
    row = frame.tail(1).iloc[0]
    return {"qvix_300etf": _float_or_none(row.get("close"))}


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {None, "", "-"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    build_macro_regime()


if __name__ == "__main__":
    main()
