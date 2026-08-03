from __future__ import annotations

import argparse
from pathlib import Path

from aquant.factors.aggregation.l3_selection import select_l3_family_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select_l3_family_models(smoke_path=args.smoke, output=args.output)


if __name__ == "__main__":
    main()
