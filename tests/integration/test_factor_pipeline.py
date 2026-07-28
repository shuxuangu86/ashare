import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from aquant.domain.data_release import DataReleaseId
from aquant.factors.aggregation.models import AlphaAggregator, ModelKind
from aquant.factors.atomic import FactorPanelInput, baseline_factor_library
from aquant.factors.evaluation.ic import forward_return_labels, information_coefficient
from aquant.factors.evaluation.quality import evaluate_quality
from aquant.factors.feature_sets import FactorMember, FeatureSetSpec
from aquant.factors.materialization import FactorMaterializationEngine, MaterializationRequest
from aquant.factors.registry import FactorRegistry
from aquant.factors.selection.clustering import correlation_matrix, hierarchical_clusters


@dataclass
class PanelLoader:
    panel: FactorPanelInput
    calls: int = 0

    def load(self, **_kwargs: object) -> FactorPanelInput:
        self.calls += 1
        return self.panel


def test_standard_pit_to_materialization_evaluation_feature_set_and_l3(
    tmp_path: Path,
) -> None:
    rows, columns = 45, 8
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(rows))
    codes = tuple(f"{index:06d}.SZ" for index in range(columns))
    time = np.arange(rows, dtype=float)[:, None]
    security = np.arange(columns, dtype=float)[None, :]
    close = 10 + 0.08 * time + 0.3 * security + np.sin((time + security) / 4)
    panel = FactorPanelInput(
        dates,
        codes,
        {
            "close": close,
            "pb": 1.2 + 0.02 * time + 0.05 * security,
            "turnover_rate": 0.5 + np.abs(np.sin(time / 5 + security)),
        },
    )
    factor_ids = {"book_to_market", "momentum_5d", "turnover_20d"}
    factors = tuple(
        factor for factor in baseline_factor_library() if factor.spec.factor_id in factor_ids
    )
    release = DataReleaseId("cn_equity_20260728_001")

    with FactorRegistry(tmp_path / "registry.sqlite3") as registry:
        for factor in factors:
            registry.register(factor.spec)
        assert {
            registry.get(factor.spec.factor_id, factor.spec.version).factor_id for factor in factors
        } == factor_ids

    loader = PanelLoader(panel)
    request = MaterializationRequest(
        data_release_id=release,
        factors=factors,
        start_date=dates[0],
        end_date=dates[-1],
        as_of_time=datetime(2026, 7, 28, tzinfo=UTC),
        universe="all_a_share",
        input_loader=loader,
        calendar=object(),
        config={"pipeline": "integration"},
        code_version="integration-test",
        config_hash=hashlib.sha256(b"integration").hexdigest(),
    )
    manifest = FactorMaterializationEngine(tmp_path / "materialized").materialize(request)
    assert manifest.total_rows == rows * columns * len(factors)
    assert loader.calls == 1

    arrays = [factor.compute_array(panel) for factor in factors]
    quality = [
        evaluate_quality(values[factor.spec.required_history - 1 :])
        for factor, values in zip(factors, arrays, strict=True)
    ]
    assert all(not item.flags for item in quality)
    labels = forward_return_labels(close, 5)
    assert any(
        np.isfinite(information_coefficient(values, labels, rank=True).mean) for values in arrays
    )

    stacked = np.column_stack([values.reshape(-1) for values in arrays])
    valid = np.all(np.isfinite(stacked), axis=1) & np.isfinite(labels.reshape(-1))
    correlations = correlation_matrix(stacked[valid])
    clusters = hierarchical_clusters(correlations, maximum_distance=0.5)
    assert len(clusters) == len(factors)

    feature_set = FeatureSetSpec(
        feature_set_id="integration_baseline",
        version="1.0.0",
        description="Deterministic integration feature set",
        target_horizon=5,
        universe="all_a_share",
        factor_members=tuple(
            FactorMember(factor_id=factor.spec.factor_id, factor_version=factor.spec.version)
            for factor in factors
        ),
        selection_method="hierarchical_clusters",
        created_from_experiment="integration_pipeline",
        data_release_id=release,
    )
    model = AlphaAggregator(ModelKind.RIDGE, alpha=1.0).fit(
        stacked[valid], labels.reshape(-1)[valid]
    )
    predictions = model.predict(stacked[valid])
    assert len(feature_set.factor_members) == len(factors)
    assert predictions.shape == (int(np.count_nonzero(valid)),)
    assert np.all(np.isfinite(predictions))
