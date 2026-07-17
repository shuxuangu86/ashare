from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TemporalFold:
    train: IndexArray
    validation: IndexArray
    test: IndexArray


@dataclass(frozen=True, slots=True)
class WalkForwardSplitter:
    train_size: int
    validation_size: int
    test_size: int
    step_size: int
    purge_size: int = 0
    embargo_size: int = 0

    def __post_init__(self) -> None:
        sizes = (self.train_size, self.validation_size, self.test_size, self.step_size)
        if any(value <= 0 for value in sizes) or self.purge_size < 0 or self.embargo_size < 0:
            raise ValueError("walk-forward sizes are invalid")

    def split(self, sample_count: int) -> tuple[TemporalFold, ...]:
        folds: list[TemporalFold] = []
        start = 0
        while True:
            train_end = start + self.train_size
            validation_start = train_end + self.purge_size
            validation_end = validation_start + self.validation_size
            test_start = validation_end + self.embargo_size
            test_end = test_start + self.test_size
            if test_end > sample_count:
                break
            folds.append(
                TemporalFold(
                    np.arange(start, train_end, dtype=np.int64),
                    np.arange(validation_start, validation_end, dtype=np.int64),
                    np.arange(test_start, test_end, dtype=np.int64),
                )
            )
            start += self.step_size
        if not folds:
            raise ValueError("not enough samples for one walk-forward fold")
        return tuple(folds)
