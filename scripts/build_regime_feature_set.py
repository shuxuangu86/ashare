#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aquant.regime import core_state_registry
from aquant.regime.definitions import MarketStateFamily, MarketStateRole, MarketStateStatus


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a version-pinned L3 regime feature set")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/regime_v1/regime_feature_sets.json"),
    )
    args = parser.parse_args()
    eligible = tuple(
        spec
        for spec in core_state_registry()
        if spec.status in {MarketStateStatus.IMPLEMENTED, MarketStateStatus.PARTIAL}
    )
    selections = {
        "market_state_raw_v1": eligible,
        "market_state_core_v1": tuple(
            spec for spec in eligible if spec.role == MarketStateRole.REGIME_INPUT
        ),
        "market_state_style_v1": tuple(
            spec for spec in eligible if spec.family == MarketStateFamily.STYLE
        ),
        "market_state_valuation_v1": tuple(
            spec for spec in eligible if spec.family == MarketStateFamily.VALUATION
        ),
        "market_state_liquidity_v1": tuple(
            spec for spec in eligible if spec.family == MarketStateFamily.LIQUIDITY
        ),
        "market_state_crowding_v1": tuple(
            spec for spec in eligible if spec.family == MarketStateFamily.CROWDING
        ),
    }
    payload = {
        "schema_version": "aquant.regime-feature-sets.v1",
        "version": "1.0.0",
        "release_status": "DRAFT",
        "feature_sets": [
            {
                "feature_set_id": feature_set_id,
                "members": [
                    {"state_id": spec.state_id, "state_version": spec.version} for spec in members
                ],
            }
            for feature_set_id, members in selections.items()
        ],
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
