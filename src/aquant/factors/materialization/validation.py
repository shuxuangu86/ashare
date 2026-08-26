import math

from aquant.factors.types import FactorResult


def validate_result(result: FactorResult) -> None:
    for value in result.values:
        if value.is_valid != (value.value is not None and math.isfinite(value.value)):
            raise ValueError("factor validity flag conflicts with value")
        if value.value is None and not value.quality_flags:
            raise ValueError("invalid factor values require a quality flag")
