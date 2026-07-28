from datetime import date


def missing_dates(
    requested: tuple[date, ...],
    materialized: tuple[date, ...],
) -> tuple[date, ...]:
    if tuple(sorted(requested)) != requested:
        raise ValueError("requested dates must be sorted")
    known = set(materialized)
    return tuple(value for value in requested if value not in known)
