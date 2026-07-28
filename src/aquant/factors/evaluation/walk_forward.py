from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeSplit:
    train: slice
    validation: slice


def walk_forward_splits(
    length: int,
    *,
    train_size: int,
    validation_size: int,
    step: int,
    expanding: bool = True,
    purge: int = 0,
    embargo: int = 0,
) -> tuple[TimeSplit, ...]:
    if min(length, train_size, validation_size, step) <= 0 or min(purge, embargo) < 0:
        raise ValueError("walk-forward sizes must be positive and gaps non-negative")
    splits: list[TimeSplit] = []
    validation_start = train_size + purge
    while validation_start + validation_size <= length:
        train_end = validation_start - purge
        train_start = 0 if expanding else max(0, train_end - train_size)
        validation_end = validation_start + validation_size
        splits.append(
            TimeSplit(slice(train_start, train_end), slice(validation_start, validation_end))
        )
        validation_start = validation_end + embargo if embargo else validation_start + step
    return tuple(splits)


def fixed_time_split(
    length: int,
    *,
    train_end: int,
    validation_end: int,
    purge: int = 0,
    embargo: int = 0,
) -> TimeSplit:
    validation_start = train_end + purge + embargo
    if not 0 < train_end <= validation_start < validation_end <= length:
        raise ValueError("fixed time split must be strictly ordered")
    return TimeSplit(slice(0, train_end), slice(validation_start, validation_end))
