"""Governed investment strategies that emit target portfolios, never broker orders."""

from aquant.strategies.discretionary import (
    DiscretionaryDecision,
    DiscretionaryFundamentalStrategy,
)

__all__ = ["DiscretionaryDecision", "DiscretionaryFundamentalStrategy"]
