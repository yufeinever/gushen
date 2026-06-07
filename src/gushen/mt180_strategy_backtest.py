from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FACTOR_DIR = Path("data/local/factor_library/mt180")
DEFAULT_CACHE_DIR = Path("data/local/guided_factor_backtests/daily_bars/qfq")
DEFAULT_OUTPUT_DIR = Path("reports/generated/mt180_strategy_backtests")
DEFAULT_LIMIT = 1000
DEFAULT_UNIVERSE_SIZE = 100
DEFAULT_HOLD_DAYS = 5
DEFAULT_COMMISSION = 0.0008
MAX_ABS_TRADE_RETURN = 3.0
MIN_SIGNAL_RATE = 0.001
MAX_SIGNAL_RATE = 0.35
MAX_FORMULA_CHARS = 6000
MAX_ASSIGNMENTS = 80

UNSUPPORTED_KEYWORDS = re.compile(
    r"\b("
    r"FINANCE|FINVALUE|DYNAINFO|WINNER|LWINNER|PWINNER|COST|INDEXC|INDEXH|INDEXL|INDEXO|INDEXV|"
    r"HY_INDEX|BKNAME|INBLOCK|EXTERNVALUE|EXTDATA_USER|SIGNALS_USER|TIME|FROMOPEN|PERIOD|"
    r"ZIG|PEAK|TROUGH|BACKSET|REFX|REFXV|CURRBARSCOUNT|ISLASTBAR|GNBLOCK"
    r")\b",
    re.IGNORECASE,
)

DRAWING_FUNCTIONS = {
    "DRAWICON",
    "DRAWTEXT",
    "DRAWNUMBER",
    "STICKLINE",
    "DRAWKLINE",
    "DRAWBAND",
    "DRAWTEXT_FIX",
    "DRAWNUMBER_FIX",
}

STYLE_SUFFIX = re.compile(
    r",(?:COLOR[A-Z0-9]+|LINETHICK\d+|NODRAW|DOTLINE|POINTDOT|STICK|VOLSTICK|CIRCLEDOT|"
    r"COLORMAGENTA|COLORYELLOW|COLORRED|COLORGREEN|COLORBLUE|COLORWHITE|COLORBLACK|COLORCYAN)"
    r".*$",
    re.IGNORECASE,
)

SIGNAL_NAME_HINT = re.compile(r"(买|选股|信号|短线|启动|突破|金叉|擒|强|红|多|进)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedStrategy:
    source_id: str
    name: str
    category: str
    indicator_type_label: str
    sales_count: int
    formula_file: str
    formula: str
    expression: str
    python_expression: str
    signal_mode: str
    assignments: int


@dataclass(frozen=True)
class StrategySummary:
    source_id: str
    name: str
    category: str
    indicator_type_label: str
    sales_count: int
    signal_mode: str
    expression: str
    tested_stocks: int
    signal_count: int
    trade_count: int
    avg_return_pct: float | None
    median_return_pct: float | None
    win_rate_pct: float | None
    best_trade_pct: float | None
    worst_trade_pct: float | None
    score: float | None
    status: str
    note: str


@dataclass(frozen=True)
class TradeRow:
    source_id: str
    name: str
    ts_code: str
    stock_name: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    net_return_pct: float


class FormulaParseError(ValueError):
    pass


class TokenStream:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def pop(self) -> str:
        token = self.peek()
        if token is None:
            raise FormulaParseError("unexpected end of expression")
        self.index += 1
        return token

    def consume(self, value: str) -> bool:
        if self.peek() == value:
            self.index += 1
            return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and backtest the first usable mt180 daily strategies.")
    parser.add_argument("--factor-dir", type=Path, default=DEFAULT_FACTOR_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--universe-size", type=int, default=DEFAULT_UNIVERSE_SIZE)
    parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS)
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    parser.add_argument("--max-trades", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    report = run_mt180_strategy_backtest(
        factor_dir=args.factor_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        universe_size=args.universe_size,
        hold_days=args.hold_days,
        commission=args.commission,
        max_trades=args.max_trades,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_mt180_strategy_backtest(
    factor_dir: Path = DEFAULT_FACTOR_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int = DEFAULT_LIMIT,
    universe_size: int = DEFAULT_UNIVERSE_SIZE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    commission: float = DEFAULT_COMMISSION,
    max_trades: int = 200_000,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((factor_dir / "manifest.json").read_text(encoding="utf-8"))
    parsed, rejected = select_and_parse_strategies(factor_dir, manifest.get("items", []), limit)
    histories = load_universe_histories(cache_dir, universe_size)
    summaries: list[StrategySummary] = []
    trades: list[TradeRow] = []

    for index, strategy in enumerate(parsed, start=1):
        returns: list[float] = []
        total_bars = 0
        tested = 0
        signal_count = 0
        strategy_trades: list[TradeRow] = []
        for ts_code, stock_name, frame in histories:
            try:
                signal = evaluate_strategy(strategy, frame)
            except Exception:
                continue
            if signal is None or signal.empty:
                continue
            tested += 1
            signal = signal.reindex(frame.index).fillna(False).astype(bool)
            signal_count += int(signal.sum())
            total_bars += len(signal)
            stock_trades = backtest_signal(
                source_id=strategy.source_id,
                name=strategy.name,
                ts_code=ts_code,
                stock_name=stock_name,
                frame=frame,
                signal=signal,
                hold_days=hold_days,
                commission=commission,
            )
            returns.extend(row.net_return_pct / 100 for row in stock_trades)
            strategy_trades.extend(stock_trades)
        signal_rate = signal_count / total_bars if total_bars else 0.0
        summaries.append(build_summary(strategy, tested, signal_count, returns, signal_rate))
        if len(trades) < max_trades:
            trades.extend(strategy_trades[: max_trades - len(trades)])
        if index % 10 == 0:
            write_progress(run_dir / "progress.json", index, len(parsed), len(trades), summaries)

    summary_path = run_dir / "strategy_summary.csv"
    trades_path = run_dir / "trades.csv"
    rewrites_path = run_dir / "rewrites.jsonl"
    rejected_path = run_dir / "rejected.jsonl"
    manifest_path = run_dir / "manifest.json"
    write_dataclass_csv(summary_path, summaries, StrategySummary)
    write_dataclass_csv(trades_path, trades, TradeRow)
    rewrites_path.write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in parsed),
        encoding="utf-8",
    )
    rejected_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejected),
        encoding="utf-8",
    )
    best = sorted(
        [item for item in summaries if item.score is not None],
        key=lambda item: item.score or -999,
        reverse=True,
    )[:20]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "factor_dir": str(factor_dir),
        "cache_dir": str(cache_dir),
        "run_dir": str(run_dir),
        "requested_limit": limit,
        "parsed_strategies": len(parsed),
        "rejected_candidates": len(rejected),
        "universe_size": len(histories),
        "hold_days": hold_days,
        "commission": commission,
        "summary_path": str(summary_path),
        "trades_path": str(trades_path),
        "rewrites_path": str(rewrites_path),
        "rejected_path": str(rejected_path),
        "top20": [asdict(item) for item in best],
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def select_and_parse_strategies(
    factor_dir: Path,
    items: list[dict[str, Any]],
    limit: int,
) -> tuple[list[ParsedStrategy], list[dict[str, Any]]]:
    ordered = sorted(items, key=lambda item: int(item.get("sales_count") or 0), reverse=True)
    parsed: list[ParsedStrategy] = []
    rejected: list[dict[str, Any]] = []
    for item in ordered:
        formula_path = factor_dir / str(item.get("formula_file", ""))
        formula = formula_path.read_text(encoding="utf-8", errors="ignore")
        source_id = str(item.get("source_id") or "")
        if UNSUPPORTED_KEYWORDS.search(formula):
            rejected.append({"source_id": source_id, "name": item.get("name"), "reason": "unsupported_data_dependency"})
            continue
        if len(formula) > MAX_FORMULA_CHARS:
            rejected.append({"source_id": source_id, "name": item.get("name"), "reason": "formula_too_long"})
            continue
        try:
            strategy = parse_strategy_item(item, formula)
            if strategy.assignments > MAX_ASSIGNMENTS:
                rejected.append({"source_id": source_id, "name": item.get("name"), "reason": "too_many_assignments"})
                continue
            parsed.append(strategy)
        except Exception as exc:
            rejected.append(
                {
                    "source_id": source_id,
                    "name": item.get("name"),
                    "reason": f"parse_failed:{type(exc).__name__}",
                    "detail": str(exc)[:300],
                }
            )
            continue
        if len(parsed) >= limit:
            break
    return parsed, rejected


def parse_strategy_item(item: dict[str, Any], formula: str) -> ParsedStrategy:
    assignments: dict[str, str] = {}
    candidates: list[tuple[str, str]] = []
    for statement in split_formula_statements(formula):
        statement = clean_statement(statement)
        if not statement:
            continue
        upper = statement.upper()
        function_name = upper.split("(", 1)[0].strip()
        if function_name in DRAWING_FUNCTIONS and "(" in statement:
            args = split_top_level(statement[statement.find("(") + 1 : statement.rfind(")")])
            if args and not has_quoted_text(args[0]):
                candidates.append(("drawing_condition", args[0]))
            continue
        target, expression, is_output = split_assignment(statement)
        if expression is None:
            continue
        target = normalize_identifier(target)
        if not target:
            continue
        assignments[target] = expression
        if is_output or SIGNAL_NAME_HINT.search(target):
            candidates.append((target, expression))
    if not candidates and assignments:
        name, expression = next(reversed(assignments.items()))
        candidates.append((name, expression))
    if not candidates:
        raise FormulaParseError("no signal expression candidate")
    expression = candidates[-1][1]
    python_expression = compile_expression(expression)
    return ParsedStrategy(
        source_id=str(item.get("source_id") or ""),
        name=str(item.get("name") or ""),
        category=str(item.get("category") or ""),
        indicator_type_label=str(item.get("indicator_type_label") or ""),
        sales_count=int(item.get("sales_count") or 0),
        formula_file=str(item.get("formula_file") or ""),
        formula=formula,
        expression=expression,
        python_expression=python_expression,
        signal_mode="boolean" if expression_has_comparison(expression) else "numeric_quantile",
        assignments=len(assignments),
    )


def evaluate_strategy(strategy: ParsedStrategy, frame: pd.DataFrame) -> pd.Series | None:
    formula = strategy.formula
    env = base_env(frame)
    for statement in split_formula_statements(formula):
        statement = clean_statement(statement)
        if not statement:
            continue
        target, expression, _ = split_assignment(statement)
        if expression is None:
            continue
        target = normalize_identifier(target)
        if not target:
            continue
        try:
            env[target.upper()] = eval_expression(expression, env)
            if target != target.upper():
                env[target] = env[target.upper()]
        except Exception:
            continue
    value = eval(strategy.python_expression, safe_eval_globals(), {"env": env})
    series = to_series(value, frame.index)
    if strategy.signal_mode == "boolean":
        return as_bool(series)
    threshold = series.rolling(120, min_periods=60).quantile(0.8)
    return (series > threshold) & (series > series.shift(1))


def backtest_signal(
    source_id: str,
    name: str,
    ts_code: str,
    stock_name: str,
    frame: pd.DataFrame,
    signal: pd.Series,
    hold_days: int,
    commission: float,
) -> list[TradeRow]:
    rows: list[TradeRow] = []
    index = 0
    last = len(frame) - 1
    while index < last - hold_days:
        if not bool(signal.iloc[index]):
            index += 1
            continue
        entry = index + 1
        exit_index = min(entry + hold_days, last)
        entry_price = float(frame.iloc[entry]["open"])
        exit_price = float(frame.iloc[exit_index]["close"])
        if entry_price <= 0 or exit_price <= 0:
            index += 1
            continue
        net_return = exit_price * (1 - commission) / (entry_price * (1 + commission)) - 1
        rows.append(
            TradeRow(
                source_id=source_id,
                name=name,
                ts_code=ts_code,
                stock_name=stock_name,
                signal_date=frame.iloc[index]["trade_date"].date().isoformat(),
                entry_date=frame.iloc[entry]["trade_date"].date().isoformat(),
                exit_date=frame.iloc[exit_index]["trade_date"].date().isoformat(),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                net_return_pct=round(max(min(net_return, MAX_ABS_TRADE_RETURN), -MAX_ABS_TRADE_RETURN) * 100, 4),
            )
        )
        index = exit_index + 1
    return rows


def build_summary(
    strategy: ParsedStrategy,
    tested: int,
    signal_count: int,
    returns: list[float],
    signal_rate: float = 0.0,
) -> StrategySummary:
    if signal_rate > MAX_SIGNAL_RATE:
        return StrategySummary(
            strategy.source_id, strategy.name, strategy.category, strategy.indicator_type_label,
            strategy.sales_count, strategy.signal_mode, strategy.expression, tested, signal_count, 0,
            None, None, None, None, None, None, "too_broad", "signal fired on too many bars"
        )
    if signal_rate < MIN_SIGNAL_RATE and signal_count > 0:
        return StrategySummary(
            strategy.source_id, strategy.name, strategy.category, strategy.indicator_type_label,
            strategy.sales_count, strategy.signal_mode, strategy.expression, tested, signal_count, 0,
            None, None, None, None, None, None, "too_sparse", "signal fired too rarely for this sampled universe"
        )
    if not returns:
        return StrategySummary(
            strategy.source_id,
            strategy.name,
            strategy.category,
            strategy.indicator_type_label,
            strategy.sales_count,
            strategy.signal_mode,
            strategy.expression,
            tested,
            signal_count,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            "no_trades",
            "parsed but produced no completed trades on the sampled universe",
        )
    avg = sum(returns) / len(returns)
    wins = sum(1 for item in returns if item > 0)
    score = avg * math.sqrt(len(returns))
    ordered = sorted(returns)
    median = ordered[len(ordered) // 2]
    return StrategySummary(
        strategy.source_id,
        strategy.name,
        strategy.category,
        strategy.indicator_type_label,
        strategy.sales_count,
        strategy.signal_mode,
        strategy.expression,
        tested,
        signal_count,
        len(returns),
        round(avg * 100, 4),
        round(median * 100, 4),
        round(wins / len(returns) * 100, 2),
        round(max(returns) * 100, 4),
        round(min(returns) * 100, 4),
        round(score, 6),
        "ok",
        "research-only formula replay on qfq daily bars",
    )


def split_formula_statements(formula: str) -> list[str]:
    formula = re.sub(r"\{.*?\}", "", formula, flags=re.DOTALL)
    formula = formula.replace("\r", "\n")
    return [part.strip() for part in formula.split(";") if part.strip()]


def clean_statement(statement: str) -> str:
    statement = statement.strip()
    statement = STYLE_SUFFIX.sub("", statement).strip()
    return statement


def split_assignment(statement: str) -> tuple[str, str | None, bool]:
    for marker, is_output in ((":=", False), (":", True)):
        if marker in statement:
            left, right = statement.split(marker, 1)
            if has_quoted_text(right):
                return left, None, is_output
            return left.strip(), right.strip(), is_output
    return statement, None, False


def split_top_level(text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    in_quote = False
    for index, character in enumerate(text):
        if character == "'":
            in_quote = not in_quote
        elif not in_quote and character == "(":
            depth += 1
        elif not in_quote and character == ")":
            depth -= 1
        elif not in_quote and character == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return args


def has_quoted_text(text: str) -> bool:
    return "'" in text or '"' in text


def normalize_identifier(name: str) -> str:
    name = re.sub(r"[^\w\u4e00-\u9fff]", "", name, flags=re.UNICODE)
    return name.upper()


def expression_has_comparison(expression: str) -> bool:
    return bool(re.search(r"(>=|<=|<>|!=|=|>|<|\bAND\b|\bOR\b)", expression, re.IGNORECASE))


def compile_expression(expression: str) -> str:
    tokens = tokenize(expression)
    stream = TokenStream(tokens)
    compiled = parse_or(stream)
    if stream.peek() is not None:
        raise FormulaParseError(f"unexpected token: {stream.peek()}")
    return compiled


def eval_expression(expression: str, env: dict[str, Any]) -> Any:
    return eval(compile_expression(expression), safe_eval_globals(), {"env": env})


def tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    pattern = re.compile(
        r"\s*(>=|<=|<>|!=|:=|[()+\-*/<>=,]|[0-9]+(?:\.[0-9]+)?|[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*|.)",
        re.UNICODE,
    )
    for match in pattern.finditer(expression):
        token = match.group(1)
        if token and not token.isspace():
            tokens.append(token.upper() if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token) else token)
    return tokens


def parse_or(stream: TokenStream) -> str:
    left = parse_and(stream)
    while stream.peek() == "OR":
        stream.pop()
        right = parse_and(stream)
        left = f"op_or({left}, {right})"
    return left


def parse_and(stream: TokenStream) -> str:
    left = parse_not(stream)
    while stream.peek() == "AND":
        stream.pop()
        right = parse_not(stream)
        left = f"op_and({left}, {right})"
    return left


def parse_not(stream: TokenStream) -> str:
    if stream.peek() == "NOT":
        stream.pop()
        return f"op_not({parse_not(stream)})"
    return parse_compare(stream)


def parse_compare(stream: TokenStream) -> str:
    left = parse_add(stream)
    while stream.peek() in {">", "<", ">=", "<=", "=", "<>", "!="}:
        op = stream.pop()
        right = parse_add(stream)
        left = f"op_cmp({left}, {right}, {op!r})"
    return left


def parse_add(stream: TokenStream) -> str:
    left = parse_mul(stream)
    while stream.peek() in {"+", "-"}:
        op = stream.pop()
        right = parse_mul(stream)
        left = f"({left} {op} {right})"
    return left


def parse_mul(stream: TokenStream) -> str:
    left = parse_unary(stream)
    while stream.peek() in {"*", "/"}:
        op = stream.pop()
        right = parse_unary(stream)
        left = f"({left} {op} {right})"
    return left


def parse_unary(stream: TokenStream) -> str:
    if stream.peek() == "-":
        stream.pop()
        return f"(-{parse_unary(stream)})"
    if stream.peek() == "+":
        stream.pop()
        return parse_unary(stream)
    return parse_primary(stream)


def parse_primary(stream: TokenStream) -> str:
    token = stream.pop()
    if token == "(":
        inner = parse_or(stream)
        if not stream.consume(")"):
            raise FormulaParseError("missing closing parenthesis")
        return f"({inner})"
    if re.match(r"^[0-9]", token):
        return token
    if re.match(r"^[A-Z_\u4e00-\u9fff]", token, re.UNICODE):
        if stream.consume("("):
            args: list[str] = []
            if not stream.consume(")"):
                while True:
                    args.append(parse_or(stream))
                    if stream.consume(")"):
                        break
                    if not stream.consume(","):
                        raise FormulaParseError("missing function comma")
            return compile_function_call(token, args)
        return f"get_var(env, {token!r})"
    raise FormulaParseError(f"bad token: {token}")


def compile_function_call(name: str, args: list[str]) -> str:
    function_map = {
        "MA": "fn_ma",
        "EMA": "fn_ema",
        "SMA": "fn_sma",
        "STD": "fn_std",
        "HHV": "fn_hhv",
        "LLV": "fn_llv",
        "REF": "fn_ref",
        "SUM": "fn_sum",
        "COUNT": "fn_count",
        "IF": "fn_if",
        "IFF": "fn_if",
        "MAX": "fn_max",
        "MIN": "fn_min",
        "ABS": "fn_abs",
        "CROSS": "fn_cross",
        "BARSLAST": "fn_barslast",
        "CONST": "fn_const",
    }
    if name not in function_map:
        raise FormulaParseError(f"unsupported function: {name}")
    return f"{function_map[name]}({', '.join(args)})"


def safe_eval_globals() -> dict[str, Any]:
    return {
        "__builtins__": {},
        "op_and": op_and,
        "op_or": op_or,
        "op_not": op_not,
        "op_cmp": op_cmp,
        "fn_ma": fn_ma,
        "fn_ema": fn_ema,
        "fn_sma": fn_sma,
        "fn_std": fn_std,
        "fn_hhv": fn_hhv,
        "fn_llv": fn_llv,
        "fn_ref": fn_ref,
        "fn_sum": fn_sum,
        "fn_count": fn_count,
        "fn_if": fn_if,
        "fn_max": fn_max,
        "fn_min": fn_min,
        "fn_abs": fn_abs,
        "fn_cross": fn_cross,
        "fn_barslast": fn_barslast,
        "fn_const": fn_const,
        "missing_series": missing_series,
        "get_var": get_var,
    }


def base_env(frame: pd.DataFrame) -> dict[str, Any]:
    env = {
        "__index__": frame.index,
        "OPEN": frame["open"],
        "O": frame["open"],
        "HIGH": frame["high"],
        "H": frame["high"],
        "LOW": frame["low"],
        "L": frame["low"],
        "CLOSE": frame["close"],
        "C": frame["close"],
        "VOL": frame["volume"],
        "V": frame["volume"],
        "VOLUME": frame["volume"],
        "AMOUNT": frame["amount"],
        "DRAWNULL": pd.Series(math.nan, index=frame.index),
    }
    return env


def missing_series(env: dict[str, Any], name: str) -> pd.Series:
    raise FormulaParseError(f"missing variable: {name}")


def get_var(env: dict[str, Any], name: str) -> Any:
    if name in env:
        return env[name]
    return missing_series(env, name)


def window(value: Any) -> int:
    if isinstance(value, pd.Series):
        value = value.dropna().iloc[-1] if not value.dropna().empty else 1
    return max(1, int(float(value)))


def to_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def as_bool(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.fillna(False).astype(bool)
    return pd.Series(bool(value))


def op_and(left: Any, right: Any) -> Any:
    return as_bool(left) & as_bool(right)


def op_or(left: Any, right: Any) -> Any:
    return as_bool(left) | as_bool(right)


def op_not(value: Any) -> Any:
    return ~as_bool(value)


def op_cmp(left: Any, right: Any, op: str) -> Any:
    if op == "=":
        return left == right
    if op in {"<>", "!="}:
        return left != right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    raise FormulaParseError(f"unsupported compare op: {op}")


def fn_ma(series: Any, n: Any) -> Any:
    return series.rolling(window(n), min_periods=window(n)).mean()


def fn_ema(series: Any, n: Any) -> Any:
    return series.ewm(span=window(n), adjust=False).mean()


def fn_sma(series: Any, n: Any, m: Any = 1) -> Any:
    return series.ewm(alpha=float(m) / window(n), adjust=False).mean()


def fn_std(series: Any, n: Any) -> Any:
    return series.rolling(window(n), min_periods=window(n)).std()


def fn_hhv(series: Any, n: Any) -> Any:
    return series.rolling(window(n), min_periods=window(n)).max()


def fn_llv(series: Any, n: Any) -> Any:
    return series.rolling(window(n), min_periods=window(n)).min()


def fn_ref(series: Any, n: Any = 1) -> Any:
    return series.shift(window(n))


def fn_sum(series: Any, n: Any) -> Any:
    return series.rolling(window(n), min_periods=window(n)).sum()


def fn_count(condition: Any, n: Any) -> Any:
    return as_bool(condition).astype(int).rolling(window(n), min_periods=window(n)).sum()


def fn_if(condition: Any, true_value: Any, false_value: Any) -> Any:
    condition = as_bool(condition)
    true_series = to_series(true_value, condition.index)
    false_series = to_series(false_value, condition.index)
    return true_series.where(condition, false_series)


def fn_max(left: Any, right: Any) -> Any:
    return pd.concat([left, right], axis=1).max(axis=1) if isinstance(left, pd.Series) else max(left, right)


def fn_min(left: Any, right: Any) -> Any:
    return pd.concat([left, right], axis=1).min(axis=1) if isinstance(left, pd.Series) else min(left, right)


def fn_abs(value: Any) -> Any:
    return value.abs() if isinstance(value, pd.Series) else abs(value)


def fn_cross(left: Any, right: Any) -> Any:
    return (left > right) & (left.shift(1) <= right.shift(1))


def fn_barslast(condition: Any) -> Any:
    condition = as_bool(condition)
    result = []
    last_seen: int | None = None
    for index, flag in enumerate(condition):
        if flag:
            last_seen = index
            result.append(0)
        elif last_seen is None:
            result.append(math.nan)
        else:
            result.append(index - last_seen)
    return pd.Series(result, index=condition.index)


def fn_const(value: Any) -> Any:
    return value


def load_universe_histories(cache_dir: Path, universe_size: int) -> list[tuple[str, str, pd.DataFrame]]:
    latest = latest_history_files(cache_dir)
    ranked: list[tuple[float, str, Path]] = []
    for ts_code, path in latest.items():
        frame = read_history_frame(path)
        if len(frame) < 420:
            continue
        ranked.append((float(frame.iloc[-1]["amount"]), ts_code, path))
    histories: list[tuple[str, str, pd.DataFrame]] = []
    for _, ts_code, path in sorted(ranked, reverse=True)[:universe_size]:
        frame = read_history_frame(path)
        stock_name = str(frame.iloc[-1].get("name") or ts_code)
        histories.append((ts_code, stock_name, frame))
    return histories


def latest_history_files(cache_dir: Path) -> dict[str, Path]:
    pattern = re.compile(r"(.+?)_\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})\.csv$")
    latest: dict[str, tuple[str, Path]] = {}
    for path in cache_dir.glob("*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        ts_code, end_date = match.groups()
        if ts_code not in latest or end_date > latest[ts_code][0]:
            latest[ts_code] = (end_date, path)
    return {key: value[1] for key, value in latest.items()}


def read_history_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close", "volume", "amount"]).reset_index(drop=True)


def write_dataclass_csv(path: Path, rows: list[Any], row_type: type[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__)  # type: ignore[attr-defined]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_progress(
    path: Path,
    completed: int,
    total: int,
    trades: int,
    summaries: list[StrategySummary],
) -> None:
    status_counts = Counter(item.status for item in summaries)
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "completed": completed,
                "total": total,
                "trades": trades,
                "status_counts": status_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
