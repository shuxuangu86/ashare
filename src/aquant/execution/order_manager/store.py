from dataclasses import dataclass, replace
from enum import StrEnum

from aquant.execution.broker_api import BrokerOrder, BrokerOrderRequest


class ExecutionRecordStatus(StrEnum):
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    request: BrokerOrderRequest
    status: ExecutionRecordStatus = ExecutionRecordStatus.APPROVED
    broker_order: BrokerOrder | None = None
    error: str | None = None


class ExecutionOrderStore:
    def __init__(self) -> None:
        self._records: dict[str, ExecutionRecord] = {}

    def approve(self, request: BrokerOrderRequest) -> ExecutionRecord:
        existing = self._records.get(request.idempotency_key)
        if existing is not None:
            if existing.request != request:
                raise ValueError("idempotency key conflicts with another request")
            return existing
        record = ExecutionRecord(request)
        self._records[request.idempotency_key] = record
        return record

    def pending(self) -> tuple[ExecutionRecord, ...]:
        return tuple(
            record
            for record in self._records.values()
            if record.status is ExecutionRecordStatus.APPROVED
        )

    def mark_submitted(self, request: BrokerOrderRequest, order: BrokerOrder) -> ExecutionRecord:
        record = replace(
            self._records[request.idempotency_key],
            status=ExecutionRecordStatus.SUBMITTED,
            broker_order=order,
        )
        self._records[request.idempotency_key] = record
        return record

    def mark_failed(self, request: BrokerOrderRequest, error: str) -> ExecutionRecord:
        record = replace(
            self._records[request.idempotency_key],
            status=ExecutionRecordStatus.FAILED,
            error=error,
        )
        self._records[request.idempotency_key] = record
        return record

    def mark_unknown(self, request: BrokerOrderRequest, error: str) -> ExecutionRecord:
        record = replace(
            self._records[request.idempotency_key],
            status=ExecutionRecordStatus.UNKNOWN,
            error=error,
        )
        self._records[request.idempotency_key] = record
        return record

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._records.values())
