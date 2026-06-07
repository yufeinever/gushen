from __future__ import annotations


class DailyDualThrustStockStrategy:
    """A minimal A-share daily-bar strategy implemented against wtpy at runtime."""

    def __new__(cls, *args, **kwargs):
        from wtpy import BaseCtaStrategy

        class _Strategy(BaseCtaStrategy):
            def __init__(
                self,
                name: str,
                code: str,
                bar_count: int,
                days: int,
                k1: float,
                k2: float,
                trade_unit: int = 100,
            ) -> None:
                BaseCtaStrategy.__init__(self, name)
                self.code = code
                self.period = "d1"
                self.bar_count = bar_count
                self.days = days
                self.k1 = k1
                self.k2 = k2
                self.trade_unit = trade_unit

            def on_init(self, context):
                context.stra_prepare_bars(self.code, self.period, self.bar_count, isMain=True)
                context.stra_log_text("DailyDualThrustStockStrategy initialized")

            def on_calculate(self, context):
                bars = context.stra_get_bars(
                    self.code,
                    self.period,
                    self.bar_count,
                    isMain=True,
                )
                if bars is None or len(bars.closes) < self.days + 2:
                    return

                closes = bars.closes
                highs = bars.highs
                lows = bars.lows
                opens = bars.opens

                hh = max(highs[-self.days - 1 : -1])
                hc = max(closes[-self.days - 1 : -1])
                ll = min(lows[-self.days - 1 : -1])
                lc = min(closes[-self.days - 1 : -1])
                trade_range = max(hh - lc, hc - ll)
                upper_bound = opens[-1] + self.k1 * trade_range
                lower_bound = opens[-1] - self.k2 * trade_range

                current_position = context.stra_get_position(self.code) / self.trade_unit
                now = int(context.stra_get_date()) * 10000 + int(context.stra_get_time())
                context.write_indicator(
                    tag=self.period,
                    time=now,
                    data={
                        "upper_bound": float(upper_bound),
                        "lower_bound": float(lower_bound),
                        "high": float(highs[-1]),
                        "low": float(lows[-1]),
                        "position_lots": float(current_position),
                    },
                )

                if current_position == 0 and highs[-1] >= upper_bound:
                    context.stra_enter_long(self.code, self.trade_unit, "enterlong")
                elif current_position > 0 and lows[-1] <= lower_bound:
                    context.stra_exit_long(self.code, self.trade_unit, "exitlong")

        return _Strategy(*args, **kwargs)

