from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from aquant.factors.selection import ConvergenceCache


def test_convergence_cache_is_disk_backed_resumable_and_content_addressed(
    tmp_path: Path,
) -> None:
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(3))
    codes = ("000001.SZ", "600000.SH")
    close = np.arange(6, dtype=float).reshape(3, 2) + 10
    cache = ConvergenceCache.create(
        tmp_path / "cache",
        factor_ids=("a", "b"),
        trade_dates=dates,
        ts_codes=codes,
        close=close,
        data_release_id="cn_equity_20260717_001",
        config_hash="a" * 64,
    )
    cache.write("a", close)
    resumed = ConvergenceCache.create(
        tmp_path / "cache",
        factor_ids=("a", "b"),
        trade_dates=dates,
        ts_codes=codes,
        close=close,
        data_release_id="cn_equity_20260717_001",
        config_hash="a" * 64,
    )
    assert set(resumed.factor_hashes) == {"a"}
    with pytest.raises(ValueError, match="incomplete"):
        resumed.finalize()
    resumed.write("b", close * 2)
    assert len(resumed.finalize()) == 64
    np.testing.assert_array_equal(resumed.close, close)


def test_convergence_cache_rejects_conflicting_factor_replay(tmp_path: Path) -> None:
    cache = ConvergenceCache.create(
        tmp_path / "cache",
        factor_ids=("a",),
        trade_dates=(date(2026, 1, 1),),
        ts_codes=("000001.SZ",),
        close=np.ones((1, 1)),
        data_release_id="cn_equity_20260717_001",
        config_hash="a" * 64,
    )
    cache.write("a", np.ones((1, 1)))
    with pytest.raises(ValueError, match="content conflict"):
        cache.write("a", np.full((1, 1), 2))


def test_convergence_cache_reopens_read_only_and_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = ConvergenceCache.create(
        root,
        factor_ids=("a",),
        trade_dates=(date(2026, 1, 1),),
        ts_codes=("000001.SZ",),
        close=np.ones((1, 1)),
        data_release_id="cn_equity_20260717_001",
        config_hash="a" * 64,
    )
    cache.write("a", np.ones((1, 1)))
    cache.finalize()
    readonly = ConvergenceCache(root, verify_content=True)
    with pytest.raises(PermissionError, match="read-only"):
        readonly.write("a", np.ones((1, 1)))

    tampered = np.load(root / "close.npy", mmap_mode="r+")
    tampered[0, 0] = 2
    tampered.flush()
    with pytest.raises(ValueError, match="close content was modified"):
        ConvergenceCache(root, verify_content=True)
