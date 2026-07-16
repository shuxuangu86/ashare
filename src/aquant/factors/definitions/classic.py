from dataclasses import dataclass

from aquant.factors.definitions.dsl import Expression, parse_expression


@dataclass(frozen=True, slots=True)
class ClassicFactorSpec:
    name: str
    version: str
    expression: Expression
    hypothesis: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip() or not self.hypothesis.strip():
            raise ValueError("classic factor metadata must not be blank")


def classic_factor_library() -> tuple[ClassicFactorSpec, ...]:
    return (
        ClassicFactorSpec(
            "momentum_20d",
            "1.0.0",
            parse_expression("RankCS(Sub(Div(Close, Ref(Close, 20)), 1))"),
            "近期相对强势可能在低频横截面上延续。",
        ),
        ClassicFactorSpec(
            "reversal_5d",
            "1.0.0",
            parse_expression("RankCS(Mul(Sub(Div(Close, Ref(Close, 5)), 1), -1))"),
            "短期超涨超跌可能产生均值回归。",
        ),
        ClassicFactorSpec(
            "volatility_20d",
            "1.0.0",
            parse_expression("RankCS(Mul(Std(Div(Close, Ref(Close, 1)), 20), -1))"),
            "低波动股票可能提供更稳定的风险调整后收益。",
        ),
        ClassicFactorSpec(
            "turnover_proxy_20d",
            "1.0.0",
            parse_expression("RankCS(Mean(Div(Amount, FloatMV), 20))"),
            "成交活跃度可能反映市场关注和流动性状态。",
        ),
    )
