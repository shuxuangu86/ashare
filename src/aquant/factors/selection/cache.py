import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt


class ConvergenceCache:
    """Disk-backed factor matrix used to keep long-history convergence bounded."""

    def __init__(
        self,
        root: Path,
        *,
        writable: bool = False,
        verify_content: bool = False,
    ) -> None:
        self.root = root
        self._writable = writable
        self.metadata_path = root / "metadata.json"
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.factor_ids = tuple(payload["factor_ids"])
        self.trade_dates = tuple(date.fromisoformat(value) for value in payload["trade_dates"])
        self.ts_codes = tuple(payload["ts_codes"])
        self.data_release_id = str(payload["data_release_id"])
        self.config_hash = str(payload["config_hash"])
        self.price_basis = str(payload["price_basis"])
        self.neutralization = str(payload["neutralization"])
        self.factor_hashes: dict[str, str] = dict(payload["factor_hashes"])
        self.close_hash = payload.get("close_hash")
        shape = (len(self.factor_ids), len(self.trade_dates), len(self.ts_codes))
        mode: Literal["r+", "r"] = "r+" if writable else "r"
        self.values = np.load(root / "factor_values.npy", mmap_mode=mode)
        self.close = np.load(root / "close.npy", mmap_mode=mode)
        if self.values.shape != shape or self.close.shape != shape[1:]:
            raise ValueError("convergence cache arrays do not match metadata shape")
        if self.values.dtype != np.float32 or self.close.dtype != np.float32:
            raise ValueError("convergence cache arrays must use float32")
        if verify_content:
            self.verify_content()

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
        price_basis: str = "ADJ_FACTOR_SCALED",
        neutralization: str = "RAW",
    ) -> "ConvergenceCache":
        if root.exists():
            existing = cls(root, writable=True, verify_content=True)
            existing._validate(
                factor_ids,
                trade_dates,
                ts_codes,
                data_release_id,
                config_hash,
                price_basis,
                neutralization,
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
        # Unwritten rows are excluded by factor_hashes, so eagerly filling a large
        # cache with NaNs only forces a full physical write before evaluation starts.
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
                "price_basis": price_basis,
                "neutralization": neutralization,
                "factor_hashes": {},
                "close_hash": _array_hash(matrix),
                "content_hash": None,
            },
        )
        return cls(root, writable=True)

    def write(self, factor_id: str, values: npt.ArrayLike) -> None:
        if not self._writable:
            raise PermissionError("convergence cache was opened read-only")
        try:
            index = self.factor_ids.index(factor_id)
        except ValueError as exc:
            raise ValueError(f"factor is not declared in convergence cache: {factor_id}") from exc
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != self.values.shape[1:]:
            raise ValueError("factor values do not align with convergence cache")
        factor_hash = _array_hash(matrix)
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
                    "price_basis": self.price_basis,
                    "neutralization": self.neutralization,
                    "close_hash": self.close_hash,
                    "factor_hashes": self.factor_hashes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._persist("PASS", content_hash)
        return content_hash

    def verify_content(self) -> None:
        if not isinstance(self.close_hash, str) or len(self.close_hash) != 64:
            raise ValueError("convergence cache has no verifiable close hash")
        if _array_hash(self.close) != self.close_hash:
            raise ValueError("convergence cache close content was modified")
        for factor_id, expected in self.factor_hashes.items():
            try:
                index = self.factor_ids.index(factor_id)
            except ValueError as exc:
                raise ValueError("convergence cache hash references unknown factor") from exc
            if _array_hash(self.values[index]) != expected:
                raise ValueError(f"convergence cache factor was modified: {factor_id}")

    def _validate(
        self,
        factor_ids: tuple[str, ...],
        trade_dates: tuple[date, ...],
        ts_codes: tuple[str, ...],
        data_release_id: str,
        config_hash: str,
        price_basis: str,
        neutralization: str,
    ) -> None:
        if (
            self.factor_ids != factor_ids
            or self.trade_dates != trade_dates
            or self.ts_codes != ts_codes
            or self.data_release_id != data_release_id
            or self.config_hash != config_hash
            or self.price_basis != price_basis
            or self.neutralization != neutralization
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
                "price_basis": self.price_basis,
                "neutralization": self.neutralization,
                "factor_hashes": self.factor_hashes,
                "close_hash": self.close_hash,
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


def _array_hash(values: npt.ArrayLike) -> str:
    array = np.asarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    if array.ndim == 0:
        digest.update(array.tobytes())
    else:
        for block in array:
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()
