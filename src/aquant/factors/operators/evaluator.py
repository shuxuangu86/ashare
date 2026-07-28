from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from aquant.factors.definitions.dsl import Expression
from aquant.factors.operators.math import safe_div, signed_power
from aquant.factors.operators.time_series import argmax, argmin, decay_linear

Array: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FactorPanel:
    symbols: tuple[str, ...]
    fields: dict[str, Array]

    def __post_init__(self) -> None:
        if not self.symbols or not self.fields:
            raise ValueError("factor panel symbols and fields must not be empty")
        shape: tuple[int, int] | None = None
        normalized: dict[str, Array] = {}
        for name, raw_values in self.fields.items():
            values = np.asarray(raw_values, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(self.symbols):
                raise ValueError("factor panel fields must have shape (time, symbols)")
            if shape is not None and values.shape != shape:
                raise ValueError("factor panel fields must share one shape")
            shape = cast(tuple[int, int], values.shape)
            normalized[name] = values.copy()
        object.__setattr__(self, "fields", normalized)

    @property
    def shape(self) -> tuple[int, int]:
        return cast(tuple[int, int], next(iter(self.fields.values())).shape)


class ExpressionEvaluator:
    def evaluate(self, expression: Expression, panel: FactorPanel) -> Array:
        return self._evaluate(expression, panel)

    def _evaluate(self, expression: Expression, panel: FactorPanel) -> Array:
        op = expression.operator
        op = {
            "Delay": "Ref",
            "Rank": "RankCS",
            "TsRank": "RankTS",
        }.get(op, op)
        if op == "Field":
            field = str(expression.arguments[0])
            try:
                return panel.fields[field].copy()
            except KeyError as exc:
                raise KeyError(f"factor panel is missing field: {field}") from exc
        args = [
            self._evaluate(value, panel)
            if isinstance(value, Expression)
            else np.full(panel.shape, value, dtype=np.float64)
            for value in expression.arguments
        ]
        if op in {"Add", "Sub", "Mul", "Div", "Power"}:
            return self._arithmetic(op, args[0], args[1])
        if op == "Log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(args[0] > 0, np.log(args[0]), np.nan)
        if op == "Abs":
            return np.abs(args[0])
        if op == "Sign":
            return np.sign(args[0])
        if op == "SignedPower":
            exponent = expression.arguments[1]
            if not isinstance(exponent, int | float):
                raise TypeError("signed-power exponent must be numeric")
            return signed_power(args[0], float(exponent))
        if op in {"Greater", "GreaterEqual", "Less", "LessEqual", "Equal", "NotEqual"}:
            return self._comparison(op, args[0], args[1])
        if op == "Where":
            return np.where(np.isfinite(args[0]), np.where(args[0] != 0, args[1], args[2]), np.nan)
        if op == "Return":
            shifted = self._shift(args[0], self._window(expression))
            return safe_div(args[0], shifted) - 1
        if op == "DecayLinear":
            return decay_linear(args[0], self._window(expression))
        if op == "ArgMax":
            return argmax(args[0], self._window(expression))
        if op == "ArgMin":
            return argmin(args[0], self._window(expression))
        if op == "Scale":
            scale = expression.arguments[1] if len(expression.arguments) == 2 else 1.0
            if not isinstance(scale, int | float):
                raise TypeError("scale target must be numeric")
            return self._scale(args[0], float(scale))
        if op in {"Ref", "Delta"}:
            window = self._window(expression)
            shifted = self._shift(args[0], window)
            return shifted if op == "Ref" else args[0] - shifted
        if op in {"Mean", "Sum", "Std", "Min", "Max", "RankTS"}:
            return self._rolling(args[0], self._window(expression), op)
        if op in {"Corr", "Cov"}:
            return self._rolling_pair(args[0], args[1], self._window(expression), op)
        if op in {"RankCS", "ZScore", "Winsorize"}:
            raw_limit = expression.arguments[1] if len(expression.arguments) == 2 else 3.0
            if not isinstance(raw_limit, int | float):  # pragma: no cover - parser guards this
                raise TypeError("winsorize limit must be numeric")
            limit = float(raw_limit)
            return self._cross_section(args[0], op, limit)
        if op == "Neutralize":
            return self._neutralize(args[0], args[1])
        raise ValueError(f"unsupported factor operator: {op}")

    @staticmethod
    def _comparison(op: str, left: Array, right: Array) -> Array:
        operations = {
            "Greater": np.greater,
            "GreaterEqual": np.greater_equal,
            "Less": np.less,
            "LessEqual": np.less_equal,
            "Equal": np.equal,
            "NotEqual": np.not_equal,
        }
        valid = np.isfinite(left) & np.isfinite(right)
        return np.where(valid, operations[op](left, right).astype(float), np.nan)

    @staticmethod
    def _scale(values: Array, target: float) -> Array:
        result = np.full(values.shape, np.nan)
        for row_number, row in enumerate(values):
            valid = np.isfinite(row)
            denominator = np.sum(np.abs(row[valid]))
            if denominator > 0:
                result[row_number, valid] = row[valid] * target / denominator
        return result

    @staticmethod
    def _arithmetic(op: str, left: Array, right: Array) -> Array:
        if op == "Add":
            return left + right
        if op == "Sub":
            return left - right
        if op == "Mul":
            return left * right
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if op == "Div":
                return np.where(right != 0, left / right, np.nan)
            return np.power(left, right)

    @staticmethod
    def _window(expression: Expression) -> int:
        value = expression.arguments[-1]
        if type(value) is not int:  # pragma: no cover - parser guarantees this
            raise TypeError("window must be an integer")
        return value

    @staticmethod
    def _shift(values: Array, periods: int) -> Array:
        result = np.full(values.shape, np.nan)
        if periods < values.shape[0]:
            result[periods:] = values[:-periods]
        return result

    @staticmethod
    def _rolling(values: Array, window: int, op: str) -> Array:
        result = np.full(values.shape, np.nan)
        for end in range(window - 1, values.shape[0]):
            sample = values[end - window + 1 : end + 1]
            if op == "Mean":
                result[end] = np.mean(sample, axis=0)
            elif op == "Sum":
                result[end] = np.sum(sample, axis=0)
            elif op == "Std":
                result[end] = np.std(sample, axis=0)
            elif op == "Min":
                result[end] = np.min(sample, axis=0)
            elif op == "Max":
                result[end] = np.max(sample, axis=0)
            else:
                result[end] = np.mean(sample <= sample[-1], axis=0)
        return result

    @staticmethod
    def _rolling_pair(left: Array, right: Array, window: int, op: str) -> Array:
        result = np.full(left.shape, np.nan)
        for end in range(window - 1, left.shape[0]):
            x = left[end - window + 1 : end + 1]
            y = right[end - window + 1 : end + 1]
            for column in range(left.shape[1]):
                if op == "Cov":
                    result[end, column] = np.cov(x[:, column], y[:, column], ddof=0)[0, 1]
                elif np.std(x[:, column]) > 0 and np.std(y[:, column]) > 0:
                    result[end, column] = np.corrcoef(x[:, column], y[:, column])[0, 1]
        return result

    @staticmethod
    def _cross_section(values: Array, op: str, limit: float) -> Array:
        result = np.full(values.shape, np.nan)
        for row_number, row in enumerate(values):
            valid = np.isfinite(row)
            sample = row[valid]
            if not sample.size:
                continue
            if op == "RankCS":
                order = np.argsort(np.argsort(sample, kind="stable"), kind="stable")
                transformed = (order + 1) / sample.size
            else:
                mean = np.mean(sample)
                std = np.std(sample)
                transformed = np.zeros_like(sample) if std == 0 else (sample - mean) / std
                if op == "Winsorize":
                    transformed = np.clip(transformed, -limit, limit)
            result[row_number, valid] = transformed
        return result

    @staticmethod
    def _neutralize(values: Array, exposure: Array) -> Array:
        result = np.full(values.shape, np.nan)
        for row_number, (target, factor) in enumerate(zip(values, exposure, strict=True)):
            valid = np.isfinite(target) & np.isfinite(factor)
            if np.count_nonzero(valid) < 2:
                continue
            design = np.column_stack((np.ones(np.count_nonzero(valid)), factor[valid]))
            coefficients = np.linalg.lstsq(design, target[valid], rcond=None)[0]
            result[row_number, valid] = target[valid] - design @ coefficients
        return result
