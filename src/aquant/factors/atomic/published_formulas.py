from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aquant.factors.atomic.models import Array, AtomicFactor, FactorPanelInput
from aquant.factors.operators.cross_sectional import cs_demean, cs_percentile
from aquant.factors.operators.math import safe_div
from aquant.factors.operators.time_series import (
    argmax,
    argmin,
    count_if,
    days_since_high,
    days_since_low,
    decay_linear,
    delay,
    delta,
    regression_residual,
    regression_slope,
    rolling_corr,
    rolling_cov,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_rank,
    rolling_std,
    rolling_sum,
    sma_cn,
)
from aquant.factors.spec import (
    FactorLayer,
    FactorRole,
    FactorSpec,
    FactorStatus,
    ImplementationStatus,
    SourceFaithfulness,
    SourceType,
)

_CONFIG_ROOT = Path(__file__).resolve().parents[4] / "configs" / "factors"
_FIELD_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}
_INDUSTRY_ALPHA101 = frozenset(
    {48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}
)
_GTJA_KNOWN_AMBIGUITIES = frozenset(
    {
        17,
        10,
        30,
        35,
        36,
        54,
        63,
        64,
        65,
        68,
        75,
        86,
        92,
        108,
        116,
        127,
        131,
        143,
        149,
        165,
        183,
        188,
    }
)
_GTJA_OCR_CORRECTIONS = frozenset(
    {
        23,
        49,
        50,
        51,
        52,
        55,
        69,
        75,
        78,
        111,
        112,
        113,
        122,
        128,
        130,
        137,
        142,
        146,
        149,
        159,
        162,
        166,
        180,
        181,
        182,
        190,
    }
)


def alpha101_original_library() -> tuple[AtomicFactor, ...]:
    return _published_library(
        _CONFIG_ROOT / "alpha101_formulas.json",
        pack="alpha101_original_v1",
    )


def gtja191_original_library() -> tuple[AtomicFactor, ...]:
    return _published_library(
        _CONFIG_ROOT / "gtja191_formulas.json",
        pack="alpha191_original_v1",
    )


def _published_library(path: Path, *, pack: str) -> tuple[AtomicFactor, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_id = str(payload["source_id"])
    return tuple(
        _build_formula_factor(
            source_id=source_id,
            pack=pack,
            source_hash=str(payload["source_pdf_sha256"]),
            formula_id=int(entry["formula_id"]),
            source_page=int(entry["source_page"]),
            raw_formula=str(entry["raw_formula"]),
        )
        for entry in payload["formulas"]
    )


def _build_formula_factor(
    *,
    source_id: str,
    pack: str,
    source_hash: str,
    formula_id: int,
    source_page: int,
    raw_formula: str,
) -> AtomicFactor:
    adopted_formula = _adopted_formula(source_id, formula_id, raw_formula)
    normalized, error = _normalize_and_validate(adopted_formula)
    faithfulness, notes = _faithfulness(source_id, formula_id, raw_formula, error)
    status = (
        ImplementationStatus.FORMULA_AMBIGUOUS
        if error is not None or "SELF" in adopted_formula.upper()
        else ImplementationStatus.IMPLEMENTED
    )
    prefix = "alpha101" if source_id == "SRC_ALPHA101" else "gtja191"
    factor_id = f"{prefix}_{formula_id:03d}"
    required_fields = _required_fields(raw_formula)
    history = _history_requirement(raw_formula)
    spec = FactorSpec(
        factor_id=factor_id,
        name=f"{prefix.upper()} #{formula_id}",
        description=f"Published formula #{formula_id} from {source_id}.",
        family="formula_alpha",
        subfamily=prefix,
        role=FactorRole.ALPHA_CANDIDATE,
        layer=FactorLayer.L2A,
        version="1.0.0",
        status=FactorStatus.DRAFT,
        hypothesis="Published short-horizon price-volume formulas may contain complementary alpha.",
        expected_direction=0,
        implementation=f"aquant.factors.atomic.published_formulas:{factor_id}",
        input_fields=required_fields,
        required_datasets=("bars_1d", "daily_basic_pit")
        if "total_market_cap" in required_fields
        else ("bars_1d",),
        required_history=history,
        minimum_periods=history,
        data_lag=1,
        availability_lag=1,
        universe="all_a_share",
        target_horizons=(1, 5, 10, 20),
        parameters={
            "factor_pack": pack,
            "formula_id": formula_id,
            "raw_formula": raw_formula,
            "adopted_formula": adopted_formula,
            "normalized_formula": normalized,
            "source_pdf_sha256": source_hash,
            "normalization_error": error,
        },
        source_type=(
            SourceType.FORMULA_LIBRARY if source_id == "SRC_ALPHA101" else SourceType.BROKER_REPORT
        ),
        source_reference=source_id,
        source_id=source_id,
        source_section="Appendix A"
        if source_id == "SRC_ALPHA101"
        else "Table 6: factor definitions",
        source_formula_id=str(formula_id),
        source_page=str(source_page),
        source_faithfulness=faithfulness,
        implementation_notes=notes,
        normalization="SOURCE_FORMULA",
        missing_policy="PRESERVE",
        warmup_policy="REQUIRE_MINIMUM_PERIODS",
        complexity_score=float(min(100, len(tuple(ast.walk(ast.parse(normalized, mode="eval"))))))
        if error is None
        else 100.0,
        implementation_status=status,
        tags=(prefix, pack, str(faithfulness), str(status), "next_session_only"),
    )

    def calculate(panel: FactorPanelInput) -> Array:
        if error is not None:
            return np.full(
                (len(panel.trade_dates), len(panel.ts_codes)),
                np.nan,
                dtype=np.float64,
            )
        return FormulaEvaluator(panel, source_id=source_id).evaluate(normalized)

    return AtomicFactor(spec, calculate)


class FormulaEvaluator:
    """Restricted AST evaluator for audited published price-volume formulas."""

    def __init__(self, panel: FactorPanelInput, *, source_id: str) -> None:
        self.panel = panel
        self.source_id = source_id
        self.shape = (len(panel.trade_dates), len(panel.ts_codes))
        self.close = self._field("close")

    def evaluate(self, expression: str) -> Array:
        value = self._node(ast.parse(expression, mode="eval").body)
        result = self._array(value)
        return np.where(np.isfinite(result), result, np.nan)

    def _node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.Attribute):
            return f"{self._node(node.value)}.{node.attr}"
        if isinstance(node, ast.UnaryOp):
            value = self._node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Not):
                return np.logical_not(value)
        if isinstance(node, ast.BinOp):
            left, right = self._node(node.left), self._node(node.right)
            return self._binary(node.op, left, right)
        if isinstance(node, ast.Compare):
            left = self._node(node.left)
            result = np.ones(self.shape, dtype=bool)
            for operation, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._node(comparator)
                result &= self._compare(operation, left, right)
                left = right
            return result
        if isinstance(node, ast.BoolOp):
            values = [np.asarray(self._node(value), dtype=bool) for value in node.values]
            reducer = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            bool_result = values[0]
            for value in values[1:]:
                bool_result = reducer(bool_result, value)
            return bool_result
        if isinstance(node, ast.IfExp):
            return np.where(
                np.asarray(self._node(node.test), dtype=bool),
                self._node(node.body),
                self._node(node.orelse),
            )
        if isinstance(node, ast.Tuple):
            return tuple(self._node(item) for item in node.elts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self._call(node.func.id, tuple(self._node(arg) for arg in node.args))
        raise ValueError(f"unsupported published formula node: {ast.dump(node)}")

    def _name(self, name: str) -> Any:
        lower = name.lower()
        if lower in _FIELD_NAMES:
            return self._field(_FIELD_NAMES[lower])
        if lower == "vol":
            return self._field("volume")
        if lower in {"ret", "returns"}:
            return safe_div(self.close, delay(self.close)) - 1
        if lower == "vwap":
            return safe_div(self._field("amount"), self._field("volume"))
        if lower == "cap":
            field = (
                "total_market_cap"
                if "total_market_cap" in self.panel.fields
                else "float_market_cap"
            )
            return self._field(field)
        if match := re.fullmatch(r"adv(\d+)", lower):
            return rolling_mean(self._field("amount"), int(match.group(1)))
        if lower in {"mkt", "mke", "benchmarkindexclose"}:
            returns = safe_div(self.close, delay(self.close)) - 1
            counts = np.sum(np.isfinite(returns), axis=1)
            market = safe_div(
                np.nansum(returns, axis=1),
                counts,
            )
            return np.broadcast_to(market[:, None], self.shape)
        if lower == "benchmarkindexopen":
            return np.zeros(self.shape, dtype=np.float64)
        if lower in {"smb", "hml"}:
            return np.zeros(self.shape, dtype=np.float64)
        if lower == "tr":
            high, low, previous = self._field("high"), self._field("low"), delay(self.close)
            return np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous)))
        if lower == "hd":
            return delta(self._field("high"))
        if lower == "ld":
            return -delta(self._field("low"))
        if lower == "dtm":
            prior_open = delay(self._field("open"))
            return np.where(
                self._field("open") <= prior_open,
                0,
                np.maximum(
                    self._field("high") - self._field("open"),
                    self._field("open") - prior_open,
                ),
            )
        if lower == "dbm":
            prior_open = delay(self._field("open"))
            return np.where(
                self._field("open") >= prior_open,
                0,
                np.maximum(
                    self._field("open") - self._field("low"),
                    self._field("open") - prior_open,
                ),
            )
        if lower == "indclass":
            return "IndClass"
        if lower == "sequence":
            return ("sequence", 20)
        if lower == "self":
            return np.full(self.shape, np.nan)
        raise ValueError(f"unknown published formula input: {name}")

    def _call(self, name: str, args: tuple[Any, ...]) -> Any:
        lower = name.lower()
        functions: dict[str, Callable[[tuple[Any, ...]], Any]] = {
            "abs": lambda x: np.abs(x[0]),
            "log": lambda x: np.log(np.where(np.asarray(x[0]) > 0, x[0], np.nan)),
            "sign": lambda x: np.sign(x[0]),
            "rank": lambda x: cs_percentile(x[0]),
            "delay": lambda x: delay(x[0], _window(x[1]) if len(x) > 1 else 1),
            "delta": lambda x: delta(x[0], _window(x[1])),
            "correlation": lambda x: rolling_corr(x[0], x[1], _window(x[2]) if len(x) > 2 else 6),
            "corr": lambda x: rolling_corr(x[0], x[1], _window(x[2]) if len(x) > 2 else 6),
            "covariance": lambda x: rolling_cov(x[0], x[1], _window(x[2])),
            "coviance": lambda x: rolling_cov(x[0], x[1], _window(x[2])),
            "sum": lambda x: rolling_sum(x[0], _window(x[1])),
            "mean": lambda x: rolling_mean(x[0], _window(x[1]) if len(x) > 1 else 12),
            "ma": lambda x: rolling_mean(x[0], _window(x[1])),
            "std": lambda x: rolling_std(x[0], _window(x[1]) if len(x) > 1 else 20, ddof=0),
            "stddev": lambda x: rolling_std(x[0], _window(x[1]) if len(x) > 1 else 20, ddof=0),
            "ts_min": lambda x: rolling_min(x[0], _window(x[1])),
            "tsmin": lambda x: rolling_min(x[0], _window(x[1])),
            "ts_max": lambda x: rolling_max(x[0], _window(x[1])),
            "tsmax": lambda x: rolling_max(x[0], _window(x[1])),
            "ts_rank": lambda x: rolling_rank(x[0], _window(x[1])),
            "tsrank": lambda x: rolling_rank(x[0], _window(x[1])),
            "ts_argmax": lambda x: argmax(x[0], _window(x[1])),
            "ts_argmin": lambda x: argmin(x[0], _window(x[1])),
            "signedpower": lambda x: _signed_power(x[0], x[1]),
            "decay_linear": lambda x: decay_linear(x[0], _window(x[1])),
            "decaylinear": lambda x: decay_linear(x[0], _window(x[1])),
            "scale": lambda x: _scale(x[0], float(x[1]) if len(x) == 2 else 1.0),
            "product": lambda x: _rolling_product(x[0], _window(x[1])),
            "prod": lambda x: _rolling_product(x[0], _window(x[1])),
            "count": lambda x: count_if(x[0], _window(x[1])),
            "sma": lambda x: sma_cn(x[0], _window(x[1]), _window(x[2]) if len(x) > 2 else 1),
            "smean": lambda x: sma_cn(x[0], _window(x[1]), _window(x[2]) if len(x) > 2 else 1),
            "wma": lambda x: _gtja_wma(x[0], _window(x[1])),
            "highday": lambda x: days_since_high(x[0], _window(x[1])),
            "lowday": lambda x: days_since_low(x[0], _window(x[1])),
            "sequence": lambda x: ("sequence", _window(x[0])),
            "regbeta": self._regbeta,
            "regresi": self._regresi,
            "sumac": lambda x: _sumac(x[0], _window(x[1]) if len(x) > 1 else None),
            "sumif": lambda x: rolling_sum(
                np.where(np.asarray(x[2], dtype=bool), x[0], 0),
                _window(x[1]),
            ),
            "indneutralize": lambda x: cs_demean(x[0]),
            "where": lambda x: np.where(np.asarray(x[0], dtype=bool), x[1], x[2]),
            "filter": lambda x: np.where(np.asarray(x[1], dtype=bool), x[0], np.nan),
        }
        if (name == "MAX" or name == "MIN") and len(args) == 1:
            operation = rolling_max if name == "MAX" else rolling_min
            return operation(args[0], 48)
        if name == "MAX" or name == "MIN":
            operation = np.maximum if name == "MAX" else np.minimum
            return operation(args[0], args[1])
        if lower == "max" and self.source_id == "SRC_ALPHA101":
            return np.fmax(args[0], args[1])
        if lower == "min" and self.source_id == "SRC_ALPHA101":
            return np.fmin(args[0], args[1])
        if lower == "max" and len(args) == 1:
            return rolling_max(args[0], 48)
        if lower == "min" and len(args) == 1:
            return rolling_min(args[0], 48)
        if lower == "max":
            return rolling_max(args[0], _window(args[1]))
        if lower == "min":
            return rolling_min(args[0], _window(args[1]))
        try:
            return functions[lower](args)
        except KeyError as exc:
            raise ValueError(f"unsupported published formula function: {name}") from exc

    def _regbeta(self, args: tuple[Any, ...]) -> Array:
        if len(args) >= 3 and not isinstance(args[1], tuple):
            window = _window(args[-1])
            minimum = max(20, window // 4)
            covariance = rolling_cov(
                args[0],
                args[1],
                window,
                min_periods=minimum,
            )
            variance = rolling_cov(
                args[1],
                args[1],
                window,
                min_periods=minimum,
            )
            return safe_div(covariance, variance)
        window = args[1][1] if len(args) == 2 and isinstance(args[1], tuple) else _window(args[-1])
        return regression_slope(args[0], window)

    def _regresi(self, args: tuple[Any, ...]) -> Array:
        return regression_residual(args[0], _window(args[-1]))

    def _binary(self, operation: ast.operator, left: Any, right: Any) -> Any:
        if isinstance(operation, ast.Add):
            return left + right
        if isinstance(operation, ast.Sub):
            return left - right
        if isinstance(operation, ast.Mult):
            return left * right
        if isinstance(operation, ast.Div):
            return safe_div(left, right)
        if isinstance(operation, ast.Pow):
            with np.errstate(all="ignore"):
                return np.power(left, right)
        if isinstance(operation, ast.BitAnd):
            return np.logical_and(left, right)
        if isinstance(operation, ast.BitOr):
            return np.logical_or(left, right)
        raise ValueError(f"unsupported binary operation: {operation!r}")

    @staticmethod
    def _compare(operation: ast.cmpop, left: Any, right: Any) -> Array:
        if isinstance(operation, ast.Lt):
            return np.asarray(left < right)
        if isinstance(operation, ast.LtE):
            return np.asarray(left <= right)
        if isinstance(operation, ast.Gt):
            return np.asarray(left > right)
        if isinstance(operation, ast.GtE):
            return np.asarray(left >= right)
        if isinstance(operation, ast.Eq):
            return np.asarray(left == right)
        if isinstance(operation, ast.NotEq):
            return np.asarray(left != right)
        raise ValueError(f"unsupported comparison: {operation!r}")

    def _field(self, name: str) -> Array:
        return np.asarray(self.panel.fields[name], dtype=np.float64)

    def _array(self, value: Any) -> Array:
        return np.asarray(np.broadcast_to(value, self.shape), dtype=np.float64)


def normalize_published_formula(formula: str) -> str:
    value = formula.strip().rstrip(";")
    value = value.replace("\u2013", "-").replace("\uff0c", ",")
    value = re.sub(r"(?<=[A-Z])\s+(?=[A-Z])", "", value)
    value = value.replace("BANCHMARK", "BENCHMARK").replace("HGIH", "HIGH")
    value = value.replace(".*", "*").replace("./", "/")
    value = re.sub(r"\bDELAT\b", "DELTA", value, flags=re.IGNORECASE)
    value = value.replace("^", "**")
    value = value.replace("||", " or ").replace("&&", " and ").replace("&", " and ")
    value = re.sub(r"\bOR\b", " or ", value)
    value = re.sub(r"(?<![<>!=])=(?!=)", "==", value)
    value = _convert_ternaries(value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_and_validate(formula: str) -> tuple[str, str | None]:
    try:
        normalized = normalize_published_formula(formula)
        ast.parse(normalized, mode="eval")
        return normalized, None
    except (SyntaxError, ValueError) as exc:
        return formula, f"{type(exc).__name__}: {exc}"


def _convert_ternaries(value: str) -> str:
    result = value
    conversions = 0
    while "?" in result:
        conversions += 1
        if conversions > 20:
            raise ValueError("formula contains too many or non-converging ternaries")
        question = result.rfind("?")
        immediate = question - 1
        while immediate >= 0 and result[immediate].isspace():
            immediate -= 1
        possible_condition_open = (
            _matching_open(result, immediate) if immediate >= 0 and result[immediate] == ")" else -1
        )
        before_condition = possible_condition_open - 1
        condition_open = (
            possible_condition_open
            if possible_condition_open >= 0
            and (
                before_condition < 0
                or not (
                    result[before_condition].isalnum() or result[before_condition] in {"_", "."}
                )
            )
            else -1
        )
        outer_open = _enclosing_open(result, question)
        opening = condition_open if condition_open >= 0 else outer_open
        if opening < 0:
            result = f"({result})"
            continue
        closing = _matching_close(
            result,
            outer_open if condition_open >= 0 and outer_open >= 0 else opening,
        )
        colon = _matching_colon(result, question, closing)
        false_end = _false_branch_end(result, colon, closing)
        condition = (
            result[opening + 1 : immediate]
            if condition_open >= 0
            else result[opening + 1 : question]
        )
        if_true = result[question + 1 : colon]
        if_false = result[colon + 1 : false_end]
        replacement = f"(where(({condition.strip()}), ({if_true.strip()}), ({if_false.strip()})))"
        suffix_start = false_end
        if condition_open < 0 and false_end == closing:
            suffix_start = closing + 1
        result = result[:opening] + replacement + result[suffix_start:]
    return result


def _enclosing_open(value: str, position: int) -> int:
    depth = 0
    for index in range(position - 1, -1, -1):
        if value[index] == ")":
            depth += 1
        elif value[index] == "(":
            if depth == 0:
                return index
            depth -= 1
    return -1


def _matching_close(value: str, opening: int) -> int:
    if opening < 0:
        return len(value)
    depth = 0
    for index in range(opening, len(value)):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("ternary expression has no closing parenthesis")


def _matching_open(value: str, closing: int) -> int:
    depth = 0
    for index in range(closing, -1, -1):
        if value[index] == ")":
            depth += 1
        elif value[index] == "(":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _false_branch_end(value: str, colon: int, closing: int) -> int:
    depth = 0
    for index in range(colon + 1, closing):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
        elif value[index] == "," and depth == 0:
            return index
    return closing


def _matching_colon(value: str, question: int, closing: int) -> int:
    nested = 0
    depth = 0
    for index in range(question + 1, closing):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
        elif value[index] == "?":
            nested += 1
        elif value[index] == ":":
            if nested:
                nested -= 1
            elif depth == 0:
                return index
    raise ValueError("ternary expression has no matching colon")


def _faithfulness(
    source_id: str,
    formula_id: int,
    formula: str,
    error: str | None,
) -> tuple[SourceFaithfulness, str]:
    if error is not None:
        return (
            SourceFaithfulness.CORRECTED_AMBIGUITY,
            f"Formula is registered but not executable: {error}",
        )
    lower = formula.lower()
    if source_id == "SRC_ALPHA101":
        if formula_id in _INDUSTRY_ALPHA101:
            return (
                SourceFaithfulness.A_SHARE_ADAPTED,
                "Industry neutralization is approximated by market demeaning pending PIT labels.",
            )
        if "vwap" in lower:
            return (
                SourceFaithfulness.NORMALIZED_EQUIVALENT,
                "VWAP is amount(CNY)/volume(shares), validated by the standard bar contract.",
            )
        return (
            SourceFaithfulness.EXACT,
            "Source formula; non-integer windows use floor as specified.",
        )
    if formula_id in _GTJA_KNOWN_AMBIGUITIES | _GTJA_OCR_CORRECTIONS:
        return (
            SourceFaithfulness.CORRECTED_AMBIGUITY,
            "Public formula contains a documented typo, ambiguous operator, "
            "or unavailable exposure.",
        )
    if any(token in lower for token in ("vwap", "mkt", "smb", "hml", "self", "benchmark")):
        return (
            SourceFaithfulness.A_SHARE_ADAPTED,
            "Uses validated VWAP derivation or an explicitly documented market-data approximation.",
        )
    return SourceFaithfulness.EXACT, "Formula transcribed from GTJA Table 6."


def _adopted_formula(source_id: str, formula_id: int, formula: str) -> str:
    if source_id != "SRC_GTJA_ALPHA191":
        return formula
    corrections = {
        23: (
            "SMA(WHERE(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)/"
            "(SMA(WHERE(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)+"
            "SMA(WHERE(CLOSE<=DELAY(CLOSE,1),STD(CLOSE,20),0),20,1))*100"
        ),
        52: (
            "SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)"
            "/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100"
        ),
        69: (
            "(SUM(DTM,20)>SUM(DBM,20)?"
            "(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):"
            "(SUM(DTM,20)=SUM(DBM,20)?0:"
            "(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))"
        ),
        166: (
            "-20*(20-1)^1.5*SUM((CLOSE/DELAY(CLOSE,1)-1-"
            "MEAN(CLOSE/DELAY(CLOSE,1)-1,20))^3,20)/"
            "((20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1)-1-"
            "MEAN(CLOSE/DELAY(CLOSE,1)-1,20))^2,20))^1.5)"
        ),
        180: (
            "WHERE(MEAN(VOLUME,20)<VOLUME,"
            "(-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)),"
            "-1*VOLUME)"
        ),
        181: (
            "SUM(((CLOSE/DELAY(CLOSE,1)-1)-"
            "MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-"
            "(BENCHMARKINDEXCLOSE-MEAN(BENCHMARKINDEXCLOSE,20))^2,20)/"
            "SUM((BENCHMARKINDEXCLOSE-MEAN(BENCHMARKINDEXCLOSE,20))^3,20)"
        ),
    }
    return corrections.get(formula_id, formula)


def _required_fields(formula: str) -> tuple[str, ...]:
    lower = formula.lower()
    fields = {field for field in _FIELD_NAMES if re.search(rf"\b{field}\b", lower)}
    derived_dependencies = {
        "tr": {"high", "low", "close"},
        "hd": {"high"},
        "ld": {"low"},
        "dtm": {"open", "high"},
        "dbm": {"open", "low"},
    }
    for name, dependencies in derived_dependencies.items():
        if re.search(rf"\b{name}\b", lower):
            fields.update(dependencies)
    if re.search(r"\bvol\b", lower):
        fields.add("volume")
    if "ret" in lower or "returns" in lower:
        fields.add("close")
    if "vwap" in lower or "adv" in lower:
        fields.update(("amount", "volume"))
    if "cap" in lower:
        fields.add("total_market_cap")
    if any(token in lower for token in ("mkt", "mke", "benchmark")):
        fields.add("close")
    return tuple(sorted(fields or {"close"}))


def _history_requirement(formula: str) -> int:
    numbers = [
        math.floor(float(item))
        for item in re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)", formula)
        if 1 <= float(item) <= 252
    ]
    return max(numbers, default=1) + 1


def _window(value: Any) -> int:
    return max(1, math.floor(float(value)))


def _scale(values: Any, target: float) -> Array:
    matrix = np.asarray(values, dtype=np.float64)
    denominator = np.nansum(np.abs(matrix), axis=1, keepdims=True)
    return safe_div(matrix * target, denominator)


def _rolling_product(values: Any, window: int) -> Array:
    matrix = np.asarray(values, dtype=np.float64)
    result = np.full(matrix.shape, np.nan)
    for row in range(window - 1, len(matrix)):
        sample = matrix[row - window + 1 : row + 1]
        valid = np.all(np.isfinite(sample), axis=0)
        result[row, valid] = np.prod(sample[:, valid], axis=0)
    return result


def _gtja_wma(values: Any, window: int) -> Array:
    matrix = np.asarray(values, dtype=np.float64)
    result = np.full(matrix.shape, np.nan)
    weights = 0.9 ** np.arange(window - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    for row in range(window - 1, len(matrix)):
        sample = matrix[row - window + 1 : row + 1]
        valid = np.all(np.isfinite(sample), axis=0)
        result[row, valid] = weights @ sample[:, valid]
    return result


def _signed_power(values: Any, exponent: Any) -> Array:
    base, power = np.broadcast_arrays(
        np.asarray(values, dtype=np.float64),
        np.asarray(exponent, dtype=np.float64),
    )
    with np.errstate(all="ignore"):
        result = np.sign(base) * np.power(np.abs(base), power)
    return np.where(np.isfinite(result), result, np.nan)


def _sumac(values: Any, window: int | None) -> Array:
    matrix = np.asarray(values, dtype=np.float64)
    if window is None:
        return np.cumsum(np.where(np.isfinite(matrix), matrix, 0), axis=0)
    return rolling_sum(matrix, window)
