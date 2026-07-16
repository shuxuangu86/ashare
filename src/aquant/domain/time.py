from datetime import UTC, datetime


def require_aware(value: datetime, *, field_name: str) -> datetime:
    """Validate an aware timestamp and normalize it to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value.astimezone(UTC)
