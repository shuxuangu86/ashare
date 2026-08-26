from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from aquant.factors.spec import FactorLayer, FactorSpec, FactorStatus, SourceType
from aquant.factors.types import FactorContext, FactorResult, FactorValue

Array = npt.NDArray[np.float64]
Calculator = Callable[["FactorPanelInput"], Array]


@dataclass(frozen=True, slots=True)
class FactorPanelInput:
    """Stable time-by-security L0 panel returned by a Standard/PIT loader."""

    trade_dates: tuple[date, ...]
    ts_codes: tuple[str, ...]
    fields: Mapping[str, Array]

    def __post_init__(self) -> None:
        expected = (len(self.trade_dates), len(self.ts_codes))
        if not self.trade_dates or not self.ts_codes:
            raise ValueError("factor input panel must not be empty")
        if tuple(sorted(self.trade_dates)) != self.trade_dates:
            raise ValueError("factor input dates must be sorted")
        if tuple(sorted(self.ts_codes)) != self.ts_codes:
            raise ValueError("factor input securities must be sorted")
        for name, values in self.fields.items():
            if not name.strip() or np.asarray(values).shape != expected:
                raise ValueError("factor input fields must match panel shape")


@dataclass(frozen=True, slots=True)
class AtomicFactor:
    spec: FactorSpec
    calculator: Calculator

    def compute_array(self, panel: FactorPanelInput) -> Array:
        missing = set(self.spec.input_fields) - panel.fields.keys()
        if missing:
            raise KeyError(f"factor input fields are missing: {sorted(missing)}")
        result = np.asarray(self.calculator(panel), dtype=np.float64)
        expected = (len(panel.trade_dates), len(panel.ts_codes))
        if result.shape != expected:
            raise ValueError(f"factor calculator returned {result.shape}, expected {expected}")
        return np.where(np.isfinite(result), result, np.nan)

    def compute(self, context: FactorContext) -> FactorResult:
        loaded = context.input_loader.load(
            fields=self.spec.input_fields,
            start_date=context.start_date,
            end_date=context.end_date,
            as_of_time=context.as_of_time,
            universe_id=context.universe_id,
            data_release_id=context.data_release_id,
        )
        if not isinstance(loaded, FactorPanelInput):
            raise TypeError("factor loader must return FactorPanelInput")
        calculated = self.compute_array(loaded)
        values = tuple(
            FactorValue(
                trade_date=trade_date,
                ts_code=ts_code,
                factor_id=self.spec.factor_id,
                factor_version=self.spec.version,
                value=float(calculated[row, column])
                if np.isfinite(calculated[row, column])
                else None,
                is_valid=bool(np.isfinite(calculated[row, column])),
                quality_flags=() if np.isfinite(calculated[row, column]) else ("NON_FINITE",),
                data_release_id=context.data_release_id,
                computed_at=context.as_of_time,
            )
            for row, trade_date in enumerate(loaded.trade_dates)
            for column, ts_code in enumerate(loaded.ts_codes)
        )
        return FactorResult(values)


def atomic_factor(
    factor_id: str,
    family: str,
    description: str,
    hypothesis: str,
    input_fields: tuple[str, ...],
    required_history: int,
    calculator: Calculator,
    *,
    expected_direction: int = 0,
    parameters: dict[str, object] | None = None,
    layer: FactorLayer = FactorLayer.L2A,
    tags: tuple[str, ...] | None = None,
    source_reference: str = "AQuant baseline factor library",
) -> AtomicFactor:
    spec = FactorSpec(
        factor_id=factor_id,
        name=factor_id,
        description=description,
        family=family,
        layer=layer,
        version="1.0.0",
        status=FactorStatus.DRAFT,
        hypothesis=hypothesis,
        expected_direction=expected_direction,
        implementation=f"aquant.factors.atomic.{family}:{factor_id}",
        input_fields=input_fields,
        required_history=required_history,
        data_lag=0,
        universe="all_a_share",
        target_horizons=(1, 5, 10, 20, 60),
        parameters=parameters or {},
        source_type=SourceType.CODE,
        source_reference=source_reference,
        tags=tags or (family, "baseline"),
        complexity_score=float(len(input_fields) + len(parameters or {})),
    )
    return AtomicFactor(spec, calculator)
