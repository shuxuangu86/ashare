from dataclasses import dataclass
from datetime import date

from aquant.domain.data_release import DataReleaseId
from aquant.ml.models.baselines import ModelKind


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    run_id: str
    model_kind: ModelKind
    data_release_id: DataReleaseId
    feature_versions: tuple[str, ...]
    training_start: date
    training_end: date
    random_seed: int
    git_hash: str
    model_checksum: str
    validation_rank_ic: float
    test_rank_ic: float

    def __post_init__(self) -> None:
        if (
            not self.run_id.strip()
            or not self.feature_versions
            or self.training_end < self.training_start
        ):
            raise ValueError("model artifact metadata is incomplete")
        if len(self.git_hash) < 7 or len(self.model_checksum) != 64:
            raise ValueError("model artifact hashes are invalid")


class ModelRegistry:
    def __init__(self) -> None:
        self._artifacts: dict[str, ModelArtifact] = {}

    def register(self, artifact: ModelArtifact) -> None:
        if artifact.run_id in self._artifacts:
            raise ValueError(f"model run already registered: {artifact.run_id}")
        self._artifacts[artifact.run_id] = artifact

    def get(self, run_id: str) -> ModelArtifact:
        return self._artifacts[run_id]
