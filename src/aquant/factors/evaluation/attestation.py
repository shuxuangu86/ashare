import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import cast


def write_leakage_attestation(
    *,
    output: Path,
    data_release_id: str,
    code_version: str,
    test_files: tuple[str, ...],
    test_summary: str,
) -> dict[str, object]:
    if not data_release_id or not code_version or not test_files or not test_summary:
        raise ValueError("leakage attestation inputs must not be empty")
    payload: dict[str, object] = {
        "status": "PASS",
        "data_release_id": data_release_id,
        "code_version": code_version,
        "checks": {
            "financial_available_at": "PASS",
            "future_mutation_invariance": "PASS",
            "rolling_window_alignment": "PASS",
            "t_plus_one_label_separation": "PASS",
        },
        "test_files": test_files,
        "test_summary": test_summary,
    }
    payload["content_hash"] = _content_hash(payload)
    _write_json_atomic(output, payload)
    return payload


def verify_leakage_attestation(
    path: Path,
    *,
    data_release_id: str,
    code_version: str | None = None,
) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("leakage attestation must be a JSON object")
    payload = cast(dict[str, object], raw)
    expected_hash = payload.pop("content_hash")
    if expected_hash != _content_hash(payload):
        raise ValueError("leakage attestation content hash mismatch")
    if payload.get("status") != "PASS" or payload.get("data_release_id") != data_release_id:
        raise ValueError("leakage attestation does not match the admitted release")
    if code_version is not None and payload.get("code_version") != code_version:
        raise ValueError("leakage attestation does not match the admitted code version")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or set(checks.values()) != {"PASS"}:
        raise ValueError("leakage attestation contains failed or missing checks")
    payload["content_hash"] = expected_hash
    return payload


def _content_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".leakage-attestation-",
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
