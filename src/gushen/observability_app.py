from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_RESULTS_ROOT = Path("reports/generated/backtesting_py")


def discover_runs(root: Path = DEFAULT_RESULTS_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.glob("*/backtesting_py_summary.json")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(summary_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary = load_json(summary_path)
    base = summary_path.parent
    equity = pd.read_csv(base / "backtesting_py_equity_curve.csv")
    trades = pd.read_csv(base / "backtesting_py_trades.csv")
    if "Date" in equity.columns:
        equity["Date"] = pd.to_datetime(equity["Date"])
    elif equity.columns[0].startswith("Unnamed"):
        equity = equity.rename(columns={equity.columns[0]: "Date"})
        equity["Date"] = pd.to_datetime(equity["Date"])
    return summary, equity, trades


def draw_equity(equity: pd.DataFrame) -> None:
    if "Equity" not in equity.columns:
        st.warning("equity curve missing Equity column")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity["Date"], y=equity["Equity"], mode="lines", name="Equity"))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Equity")
    st.plotly_chart(fig, use_container_width=True)


def draw_drawdown(equity: pd.DataFrame) -> None:
    if "Equity" not in equity.columns:
        return
    peak = equity["Equity"].cummax()
    drawdown = equity["Equity"] / peak - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity["Date"], y=drawdown, fill="tozeroy", name="Drawdown"))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)


def draw_trade_returns(trades: pd.DataFrame) -> None:
    if trades.empty or "ReturnPct" not in trades.columns:
        st.info("no trade return series")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(trades) + 1)), y=trades["ReturnPct"], name="Trade Return"))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)


def render_app(results_root: Path = DEFAULT_RESULTS_ROOT) -> None:
    st.set_page_config(page_title="Gushen Backtest Observatory", layout="wide")
    st.title("Gushen Backtest Observatory")

    runs = discover_runs(results_root)
    if not runs:
        st.warning(f"No backtest runs found under {results_root}")
        return

    labels = [str(path.parent.relative_to(results_root)) for path in runs]
    selected = st.sidebar.selectbox("Backtest run", labels)
    summary_path = runs[labels.index(selected)]
    summary, equity, trades = load_run(summary_path)
    stats = summary.get("stats", {})

    st.caption(f"{summary.get('engine')} / {summary.get('strategy')} / {summary.get('start')} to {summary.get('end')}")

    cols = st.columns(5)
    metrics = [
        ("Return", "Return [%]", "{:.2f}%"),
        ("Buy & Hold", "Buy & Hold Return [%]", "{:.2f}%"),
        ("Max DD", "Max. Drawdown [%]", "{:.2f}%"),
        ("Trades", "# Trades", "{:.0f}"),
        ("Win Rate", "Win Rate [%]", "{:.2f}%"),
    ]
    for col, (label, key, fmt) in zip(cols, metrics):
        value = stats.get(key)
        col.metric(label, "n/a" if value is None else fmt.format(float(value)))

    tab_equity, tab_trades, tab_stats, tab_files = st.tabs(["Equity", "Trades", "Stats", "Files"])
    with tab_equity:
        draw_equity(equity)
        draw_drawdown(equity)
    with tab_trades:
        draw_trade_returns(trades)
        st.dataframe(trades, use_container_width=True)
    with tab_stats:
        st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "value"}), use_container_width=True)
    with tab_files:
        st.json(
            {
                "summary": str(summary_path),
                "panel": summary.get("panel_path"),
                "equity": summary.get("equity_path"),
                "trades": summary.get("trades_path"),
                "stats": summary.get("stats_path"),
            }
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Streamlit backtest observability app.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args, _ = parser.parse_known_args(argv)
    render_app(Path(args.results_root))


if __name__ == "__main__":
    main()
