from datetime import date, timedelta

import numpy as np

from aquant.strategies.microcap.experiments import RebalanceFrequency
from aquant.strategies.nested_l4 import select_l4_configuration


def test_l4_configuration_uses_only_supplied_inner_validation() -> None:
    generator = np.random.default_rng(12)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(80))
    scores = generator.normal(size=(80, 40))
    close = np.full((80, 40), 100.0)
    returns = 0.01 * np.sign(scores) + generator.normal(scale=0.002, size=scores.shape)
    for index in range(1, len(close)):
        close[index] = close[index - 1] * (1 + returns[index - 1])
    validation = np.arange(20, 60, dtype=np.int64)
    first = select_l4_configuration(
        scores=scores,
        close=close,
        trade_dates=dates,
        validation_positions=validation,
        target_counts=(10, 20),
        frequencies=(RebalanceFrequency.WEEKLY, RebalanceFrequency.MONTHLY),
    )
    changed = close.copy()
    changed[60:] *= 20
    second = select_l4_configuration(
        scores=scores,
        close=changed,
        trade_dates=dates,
        validation_positions=validation,
        target_counts=(10, 20),
        frequencies=(RebalanceFrequency.WEEKLY, RebalanceFrequency.MONTHLY),
    )

    assert first == second
    assert first["outer_test_used_for_selection"] is False
    assert len(first["candidates"]) == 4
