import json
from pathlib import Path

import pytest

from aquant.cli import main

PROJECT_ROOT = Path(__file__).parents[2]


def test_config_check_prints_safe_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["config-check", "--config", str(PROJECT_ROOT / "config" / "base.yaml")])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["mode"] == "BACKTEST"
    assert output["database"]["password"] == "**********"
