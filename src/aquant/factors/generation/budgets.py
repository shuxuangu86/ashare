DEFAULT_FAMILY_BUDGETS = {
    "size_value": 50,
    "quality_growth": 100,
    "investment_accruals": 80,
    "momentum_reversal": 120,
    "volatility_liquidity": 100,
    "price_volume": 200,
    "events_behavior": 150,
    "microcap_risk": 100,
}


def enforce_budget(family: str, count: int, budgets: dict[str, int] | None = None) -> None:
    configured = budgets or DEFAULT_FAMILY_BUDGETS
    if family not in configured:
        raise ValueError(f"candidate family has no configured budget: {family}")
    if count < 0 or count > configured[family]:
        raise ValueError(f"candidate batch exceeds {family} budget")
