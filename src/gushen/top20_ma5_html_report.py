from __future__ import annotations

import argparse
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from gushen.top20_ma5_pullback_strategy import (
    StrategyConfig,
    build_market_frame,
    load_daily_frames,
)


DEFAULT_BACKTEST_DIR = Path("reports/generated/top20_ma5_pullback_last_month")
DEFAULT_OUTPUT_PATH = Path("reports/generated/top20_ma5_pullback_last_month/report.html")


@dataclass(frozen=True)
class ReportConfig:
    start_date: str
    end_date: str
    cache_dir: Path
    backtest_dir: Path
    output_path: Path
    pages_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML decision report for Top20 MA5 pullback backtests.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/local/guided_factor_backtests/daily_bars/qfq"))
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_BACKTEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pages-dir", type=Path, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_payload = read_json(args.backtest_dir / "top20_ma5_summary.json")
    summary = summary_payload.get("summary", summary_payload)
    config = ReportConfig(
        start_date=args.start_date or str(summary["start_date"]),
        end_date=args.end_date or str(summary["end_date"]),
        cache_dir=args.cache_dir,
        backtest_dir=args.backtest_dir,
        output_path=args.output,
        pages_dir=args.pages_dir or args.output.with_name(f"{args.output.stem}_pages"),
    )
    output = build_report(config)
    print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))


def build_report(config: ReportConfig) -> Path:
    strategy_config = StrategyConfig(start_date=config.start_date, end_date=config.end_date)
    frames = load_daily_frames(config.cache_dir, strategy_config)
    market = build_market_frame(frames, strategy_config)
    top20 = market[market["amount_rank"] <= 20].copy()
    candidates = read_csv(config.backtest_dir / "top20_ma5_candidates.csv")
    trades = read_csv(config.backtest_dir / "top20_ma5_trades.csv")
    portfolio = read_csv(config.backtest_dir / "top20_ma5_portfolio.csv")
    summary_payload = read_json(config.backtest_dir / "top20_ma5_summary.json")
    summary = summary_payload.get("summary", summary_payload)
    if config.pages_dir.exists():
        shutil.rmtree(config.pages_dir)
    config.pages_dir.mkdir(parents=True, exist_ok=True)

    selected_keys = set(zip(candidates["signal_date"], candidates["code"])) if not candidates.empty else set()
    trade_keys = set(zip(trades["signal_date"], trades["code"])) if not trades.empty else set()
    filled = portfolio[portfolio["status"] == "filled"].copy() if not portfolio.empty else pd.DataFrame()
    if not filled.empty and not trades.empty:
        filled = filled.merge(
            trades[
                [
                    "signal_date",
                    "entry_date",
                    "exit_date",
                    "code",
                    "entry_ma5",
                    "exit_reason",
                    "next_day_high",
                    "next_day_close",
                ]
            ],
            on=["signal_date", "entry_date", "exit_date", "code"],
            how="left",
        )
    filled_keys = set(zip(filled["signal_date"], filled["code"])) if not filled.empty else set()

    date_links: list[dict[str, Any]] = []
    date_groups = list(top20.sort_values(["trade_date", "amount_rank"]).groupby("trade_date"))
    date_filenames = {str(trade_date): f"day_{trade_date}.html" for trade_date, _ in date_groups}
    latest_report_date = str(date_groups[-1][0]) if date_groups else ""
    for index, (trade_date, group) in enumerate(date_groups):
        rows = []
        selected_count = 0
        filled_count = 0
        for row in group.to_dict("records"):
            key = (str(row["trade_date"]), str(row["code"]))
            selected = key in selected_keys
            has_pullback = key in trade_keys
            is_filled = key in filled_keys
            pending_next_day = selected and str(row["trade_date"]) == latest_report_date and not has_pullback
            selected_count += int(selected)
            filled_count += int(is_filled)
            reason = decision_reason(row, selected, has_pullback, is_filled, pending_next_day)
            rows.append(
                "<tr>"
                f"<td>{int(row['amount_rank'])}</td>"
                f"<td>{escape(row['code'])}</td>"
                f"<td>{escape(row.get('name', ''))}</td>"
                f"<td>{format_number(row['amount'] / 100000000, 2)}</td>"
                f"<td>{format_number(row['close'], 2)}</td>"
                f"<td>{format_number(row['ma5'], 2)}</td>"
                f"<td><span class=\"badge {badge_class(selected, has_pullback, is_filled)}\">{escape(reason)}</span></td>"
                "</tr>"
            )
        filename = date_filenames[str(trade_date)]
        prev_link = date_filenames.get(str(date_groups[index - 1][0])) if index > 0 else None
        next_link = date_filenames.get(str(date_groups[index + 1][0])) if index + 1 < len(date_groups) else None
        page_body = "\n".join(
            [
                page_nav("../" + config.output_path.name, prev_link, next_link),
                section(
                    f"{trade_date} 选股决策",
                    f"Top20: {len(group)}；入选: {selected_count}；组合实际成交: {filled_count}。",
                    table_html(
                        ["成交额排名", "代码", "名称", "成交额(亿)", "收盘", "MA5", "决策"],
                        rows,
                    ),
                ),
            ]
        )
        (config.pages_dir / filename).write_text(
            html_document(page_body, include_plotly=False),
            encoding="utf-8",
        )
        date_links.append(
            {
                "date": str(trade_date),
                "href": f"{config.pages_dir.name}/{filename}",
                "top20": len(group),
                "selected": selected_count,
                "filled": filled_count,
            }
        )

    trade_links: list[dict[str, Any]] = []
    filled_records = filled.sort_values(["entry_date", "code"]).to_dict("records")
    trade_filenames = [f"trade_{index + 1:02d}_{trade['code']}_{trade['entry_date']}.html" for index, trade in enumerate(filled_records)]
    for index, trade in enumerate(filled_records):
        code = str(trade["code"])
        frame = frames.get(code)
        if frame is None:
            continue
        chart_frame = chart_window(frame, str(trade["signal_date"]), str(trade["exit_date"]))
        filename = trade_filenames[index]
        prev_link = trade_filenames[index - 1] if index > 0 else None
        next_link = trade_filenames[index + 1] if index + 1 < len(trade_filenames) else None
        page_body = "\n".join(
            [
                page_nav("../" + config.output_path.name, prev_link, next_link),
                section(
                    f"{trade['entry_date']} -> {trade['exit_date']} {code} {trade.get('name', '')}",
                    trade_summary_line(trade),
                    chart_html(chart_frame, trade),
                ),
            ]
        )
        (config.pages_dir / filename).write_text(
            html_document(page_body, include_plotly=True),
            encoding="utf-8",
        )
        trade_links.append(
            {
                "href": f"{config.pages_dir.name}/{filename}",
                "entry_date": str(trade["entry_date"]),
                "exit_date": str(trade["exit_date"]),
                "code": code,
                "name": str(trade.get("name", "")),
                "net_return_pct": trade.get("net_return_pct"),
            }
        )

    body = "\n".join(
        [
            header_html(summary, config),
            section("数据口径", data_notes(), ""),
            section("实际成交 K 线分页", "每笔成交一页，图上标出了信号日、买点和卖点。", link_table(
                ["买入", "卖出", "代码", "名称", "净收益", "页面"],
                [
                    [
                        item["entry_date"],
                        item["exit_date"],
                        item["code"],
                        item["name"],
                        f"{format_number(item['net_return_pct'], 2)}%",
                        f"<a href=\"{escape(item['href'])}\">查看K线</a>",
                    ]
                    for item in trade_links
                ],
            )),
            section("逐日选股决策分页", "每个交易日一页，避免一次性加载全部表格和图表。", link_table(
                ["日期", "Top20", "入选", "实际成交", "页面"],
                [
                    [
                        item["date"],
                        item["top20"],
                        item["selected"],
                        item["filled"],
                        f"<a href=\"{escape(item['href'])}\">查看选股</a>",
                    ]
                    for item in date_links
                ],
            )),
        ]
    )
    document = html_document(body, include_plotly=False)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(document, encoding="utf-8")
    return config.output_path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_reason(
    row: dict[str, Any],
    selected: bool,
    has_pullback: bool,
    is_filled: bool,
    pending_next_day: bool = False,
) -> str:
    buy_price = format_number(row["ma5"], 2)
    if is_filled:
        return f"实际成交: 回踩到{buy_price}买入"
    if has_pullback:
        return f"触发回踩到{buy_price}，组合跳过"
    if pending_next_day:
        return f"入选，次日回踩到{buy_price}买入"
    if selected:
        return f"入选但未回踩{buy_price}"
    if float(row["close"]) <= float(row["ma5"]):
        return "未入选: 收盘未站上MA5"
    return "未入选"


def badge_class(selected: bool, has_pullback: bool, is_filled: bool) -> str:
    if is_filled:
        return "filled"
    if has_pullback:
        return "trade"
    if selected:
        return "selected"
    return "rejected"


def chart_window(frame: pd.DataFrame, signal_date: str, exit_date: str) -> pd.DataFrame:
    dates = frame["trade_date"].tolist()
    signal_index = dates.index(signal_date)
    exit_index = dates.index(exit_date)
    start = max(0, signal_index - 8)
    end = min(len(frame), exit_index + 5)
    return frame.iloc[start:end].copy()


def chart_html(frame: pd.DataFrame, trade: dict[str, Any]) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame["trade_date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="K线",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["trade_date"],
            y=frame["ma5"],
            mode="lines",
            line={"color": "#2563eb", "width": 1.5},
            name="MA5",
        )
    )
    marker_points = [
        ("信号", trade["signal_date"], trade.get("entry_ma5", trade["entry_price"]), "#64748b", "circle"),
        ("买点", trade["entry_date"], trade["entry_price"], "#16a34a", "triangle-up"),
        ("卖点", trade["exit_date"], trade["exit_price"], "#dc2626", "triangle-down"),
    ]
    for name, x_value, y_value, color, symbol in marker_points:
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[float(y_value)],
                mode="markers+text",
                marker={"size": 12, "color": color, "symbol": symbol},
                text=[name],
                textposition="top center",
                name=name,
            )
        )
    fig.update_layout(
        height=420,
        margin={"l": 30, "r": 20, "t": 20, "b": 30},
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def header_html(summary: dict[str, Any], config: ReportConfig) -> str:
    cards = [
        ("窗口", f"{config.start_date} 至 {config.end_date}"),
        ("候选", summary.get("candidates")),
        ("实际成交", summary.get("filled_trades")),
        ("胜率", f"{summary.get('win_rate_pct')}%"),
        ("组合收益", f"{summary.get('portfolio_return_pct')}%"),
        ("最大回撤", f"{summary.get('max_drawdown_pct')}%"),
    ]
    items = "".join(f"<div class=\"metric\"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>" for k, v in cards)
    return f"<header><h1>Top20 MA5 回踩超短线回测报告</h1><p>日线代理版本，使用 Plotly 开源 K 线图。</p><div class=\"metrics\">{items}</div></header>"


def data_notes() -> str:
    return (
        "本报告基于日线缓存生成，只能近似视频策略。"
        "无法真实验证 10 点前/14:30 后成交、盘口排队、涨停封单、盘中先后顺序和真实冲高卖点；"
        "因此结果用于研究验证，不作为实盘建议。"
    )


def trade_summary_line(trade: dict[str, Any]) -> str:
    return (
        f"信号日 {trade['signal_date']}，买入价 {format_number(trade['entry_price'], 3)}，"
        f"卖出价 {format_number(trade['exit_price'], 3)}，净收益 {format_number(trade['net_return_pct'], 2)}%，"
        f"PnL {format_number(trade['pnl'], 2)}。"
    )


def section(title: str, intro: str, content: str) -> str:
    return f"<section><h2>{escape(title)}</h2><p>{escape(intro)}</p>{content}</section>"


def table_html(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th>{escape(item)}</th>" for item in headers)
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def link_table(headers: list[str], rows: list[list[Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append("<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>")
    return table_html(headers, table_rows)


def page_nav(index_href: str, prev_href: str | None, next_href: str | None) -> str:
    prev_html = f"<a href=\"{escape(prev_href)}\">上一页</a>" if prev_href else "<span>上一页</span>"
    next_html = f"<a href=\"{escape(next_href)}\">下一页</a>" if next_href else "<span>下一页</span>"
    return f"<nav><a href=\"{escape(index_href)}\">目录</a>{prev_html}{next_html}</nav>"


def html_document(body: str, include_plotly: bool = False) -> str:
    plotly_script = '  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>' if include_plotly else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top20 MA5 回踩超短线回测报告</title>
{plotly_script}
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    header {{ padding: 28px 36px; background: #111827; color: white; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0 0 14px; color: #64748b; line-height: 1.6; }}
    header p {{ color: #cbd5e1; }}
    section {{ margin: 18px auto; padding: 22px; max-width: 1280px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric {{ padding: 12px; background: #1f2937; border-radius: 8px; }}
    .metric span {{ display: block; color: #cbd5e1; font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; white-space: nowrap; }}
    th {{ background: #f1f5f9; color: #334155; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; }}
    .filled {{ background: #dcfce7; color: #166534; }}
    .trade {{ background: #dbeafe; color: #1d4ed8; }}
    .selected {{ background: #fef3c7; color: #92400e; }}
    .rejected {{ background: #f1f5f9; color: #475569; }}
    nav {{ display: flex; gap: 10px; max-width: 1280px; margin: 16px auto 0; padding: 0 22px; }}
    nav a, nav span {{ padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; color: #1f2937; text-decoration: none; }}
    nav span {{ color: #94a3b8; }}
    a {{ color: #2563eb; }}
    @media (max-width: 900px) {{ .metrics {{ grid-template-columns: repeat(2, 1fr); }} section {{ margin: 12px; padding: 16px; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def escape(value: Any) -> str:
    return html.escape(str(value))


def format_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
