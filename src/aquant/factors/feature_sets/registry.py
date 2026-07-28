import os
import tempfile
from pathlib import Path

from aquant.factors.feature_sets.spec import FeatureSetSpec


class FeatureSetRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def publish(self, spec: FeatureSetSpec) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{spec.feature_set_id}-{spec.version}.json"
        if target.exists():
            existing = FeatureSetSpec.model_validate_json(target.read_text(encoding="utf-8"))
            if existing.content_hash != spec.content_hash:
                raise ValueError("feature-set version already exists with different content")
            return target
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".feature-set-",
            suffix=".json",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(spec.model_dump_json(indent=2) + "\n")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target
