"""Windows execution-agent entry point.

The concrete QMT gateway must be supplied by the broker-specific installation. Importing this
module never enables LIVE submission.
"""

from aquant.config import load_settings


def main() -> int:
    settings = load_settings(
        (
            "config/base.yaml",
            "config/data.yaml",
            "config/risk.yaml",
            "config/brokers/qmt.yaml",
        )
    )
    if settings.live_trading.enabled:
        raise RuntimeError(
            "execution agent refuses to unlock LIVE without runtime approval service"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
