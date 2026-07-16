from collections.abc import Iterable
from datetime import date
from decimal import Decimal

from aquant.data.quality.report import QualityIssue, QualityReport, Severity
from aquant.domain.market_data import DailyBar, SecurityStatus

BarKey = tuple[str, date]


class DailyBarQualityGate:
    def validate(
        self,
        bars: Iterable[DailyBar],
        *,
        open_dates: Iterable[date] | None = None,
        statuses: Iterable[SecurityStatus] = (),
    ) -> QualityReport:
        resolved = tuple(bars)
        issues: list[QualityIssue] = []
        keys: set[BarKey] = set()
        for bar in resolved:
            key = (bar.symbol.canonical, bar.trade_date)
            if key in keys:
                issues.append(self._issue("DUPLICATE_PRIMARY_KEY", key, "duplicate daily bar"))
            keys.add(key)
        if open_dates is not None:
            valid_dates = set(open_dates)
            for symbol, trade_date in keys:
                if trade_date not in valid_dates:
                    issues.append(
                        self._issue(
                            "NON_TRADING_DATE",
                            (symbol, trade_date),
                            "bar date is not an open session",
                        )
                    )
        status_by_key = {(str(item.symbol), item.trade_date): item for item in statuses}
        for key in keys:
            status = status_by_key.get(key)
            if status is not None and status.suspended:
                issues.append(self._issue("BAR_ON_SUSPENSION", key, "suspended security has a bar"))
        return QualityReport("daily_bars", len(resolved), tuple(issues))

    def compare_sources(
        self,
        primary: Iterable[DailyBar],
        validation: Iterable[DailyBar],
        *,
        close_relative_tolerance: Decimal = Decimal("0.001"),
        volume_relative_tolerance: Decimal = Decimal("0.02"),
    ) -> QualityReport:
        if close_relative_tolerance < 0 or volume_relative_tolerance < 0:
            raise ValueError("source comparison tolerances must be non-negative")
        left = self._index(primary)
        right = self._index(validation)
        issues: list[QualityIssue] = []
        all_keys = set(left) | set(right)
        for key in sorted(all_keys):
            if key not in left or key not in right:
                issues.append(
                    self._issue("CROSS_SOURCE_MISSING", key, "record missing in one source")
                )
                continue
            primary_bar = left[key]
            validation_bar = right[key]
            if (
                self._relative_difference(primary_bar.close, validation_bar.close)
                > close_relative_tolerance
            ):
                issues.append(self._issue("CLOSE_MISMATCH", key, "close differs beyond tolerance"))
            if (
                self._relative_difference(primary_bar.volume, validation_bar.volume)
                > volume_relative_tolerance
            ):
                issues.append(
                    self._issue("VOLUME_MISMATCH", key, "volume differs beyond tolerance")
                )
        return QualityReport("daily_bars.cross_source", len(all_keys), tuple(issues))

    @staticmethod
    def _index(bars: Iterable[DailyBar]) -> dict[BarKey, DailyBar]:
        result: dict[BarKey, DailyBar] = {}
        for bar in bars:
            key = (bar.symbol.canonical, bar.trade_date)
            if key in result:
                raise ValueError(f"duplicate source comparison key: {key}")
            result[key] = bar
        return result

    @staticmethod
    def _relative_difference(left: Decimal, right: Decimal) -> Decimal:
        denominator = max(abs(left), abs(right), Decimal("1e-18"))
        return abs(left - right) / denominator

    @staticmethod
    def _issue(code: str, key: BarKey, message: str) -> QualityIssue:
        return QualityIssue(Severity.ERROR, code, f"{key[0]}@{key[1].isoformat()}", message)
