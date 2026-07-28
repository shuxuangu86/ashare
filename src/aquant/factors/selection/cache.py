import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt


class ConvergenceCache:
    """Disk-backed factor matrix used to keep long-history convergence bounded."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata_path = root / "metadata.json"
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.factor_ids = tuple(payload["factor_ids"])
        self.trade_dates = tuple(date.fromisoformat(value) for value in payload["trade_dates"])
        self.ts_codes = tuple(payload["ts_codes"])
        self.data_release_id = str(payload["data_release_id"])
        self.config_hash = str(payload["config_hash"])
        self.factor_hashes: dict[str, str] = dict(payload["factor_hashes"])
        shape = (len(self.factor_ids), len(self.trade_dates), len(self.ts_codes))
        self.values = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            root / "factor_values.npy",
            mode="r+",
            dtype=np.float32,
            shape=shape,
        )
        self.close = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            root / "close.npy",
            mode="r+",
            dtype=np.float32,
            shape=shape[1:],
        )

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        factor_ids: tuple[str, ...],
        trade_dates: tuple[date, ...],
        ts_codes: tuple[str, ...],
        close: npt.ArrayLike,
        data_release_id: str,
        config_hash: str,
    ) -> "ConvergenceCache":
        if root.exists():
            existing = cls(root)
            existing._validate(
                factor_ids,
                trade_dates,
                ts_codes,
                data_release_id,
                config_hash,
            )
            return existing
        if not factor_ids or len(set(factor_ids)) != len(factor_ids):
            raise ValueError("convergence cache factor ids must be unique")
        matrix = np.asarray(close, dtype=np.float32)
        expected = (len(trade_dates), len(ts_codes))
        if matrix.shape != expected:
            raise ValueError("convergence cache close matrix does not align")
        root.mkdir(parents=True)
        values = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            root / "factor_values.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(factor_ids), *expected),
        )
        values[:] = np.nan
        values.flush()
        close_values = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            root / "close.npy",
            mode="w+",
            dtype=np.float32,
            shape=expected,
        )
        close_values[:] = matrix
        close_values.flush()
        _write_metadata(
            root / "metadata.json",
            {
                "status": "RUNNING",
                "factor_ids": factor_ids,
                "trade_dates": tuple(value.isoformat() for value in trade_dates),
                "ts_codes": ts_codes,
                "data_release_id": data_release_id,
                "config_hash": config_hash,
                "factor_hashes": {},
                "content_hash": None,
            },
        )
        return cls(root)

    def write(self, factor_id: str, values: npt.ArrayLike) -> None:
        try:
            index = self.factor_ids.index(factor_id)
        except ValueError as exc:
            raise ValueError(f"factor is not declared in convergence cache: {factor_id}") from exc
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != self.values.shape[1:]:
            raise ValueError("factor values do not align with convergence cache")
        factor_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
        existing = self.factor_hashes.get(factor_id)
        if existing is not None and existing != factor_hash:
            raise ValueError("convergence cache factor content conflict")
        self.values[index] = matrix
        self.values.flush()
        self.factor_hashes[factor_id] = factor_hash
        self._persist("RUNNING")

    def finalize(self) -> str:
        missing = set(self.factor_ids) - set(self.factor_hashes)
        if missing:
            raise ValueError(f"convergence cache is incomplete: {sorted(missing)}")
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "config_hash": self.config_hash,
                    "data_release_id": self.data_release_id,
                    "factor_hashes": self.factor_hashes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._persist("PASS", content_hash)
        return content_hash

    def _validate(
        self,
        factor_ids: tuple[str, ...],
        trade_dates: tuple[date, ...],
        ts_codes: tuple[str, ...],
        data_release_id: str,
        config_hash: str,
    ) -> None:
        if (
            self.factor_ids != factor_ids
            or self.trade_dates != trade_dates
            or self.ts_codes != ts_codes
            or self.data_release_id != data_release_id
            or self.config_hash != config_hash
        ):
            raise ValueError("existing convergence cache does not match evaluation")

    def _persist(self, status: str, content_hash: str | None = None) -> None:
        _write_metadata(
            self.metadata_path,
            {
                "status": status,
                "factor_ids": self.factor_ids,
                "trade_dates": tuple(value.isoformat() for value in self.trade_dates),
                "ts_codes": self.ts_codes,
                "data_release_id": self.data_release_id,
                "config_hash": self.config_hash,
                "factor_hashes": self.factor_hashes,
                "content_hash": content_hash,
            },
        )


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".convergence-",
        suffix=".json",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
