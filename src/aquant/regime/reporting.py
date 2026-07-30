from __future__ import annotations

from collections.abc import Iterable


def correlation_clusters(
    correlations: Iterable[tuple[str, str, int, float]],
    *,
    threshold: float = 0.95,
) -> dict[str, tuple[str, ...]]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("correlation threshold must be in (0, 1]")
    graph: dict[str, set[str]] = {}
    for left, right, _observations, correlation in correlations:
        graph.setdefault(left, set())
        graph.setdefault(right, set())
        if left != right and abs(correlation) >= threshold:
            graph[left].add(right)
            graph[right].add(left)
    output: dict[str, tuple[str, ...]] = {}
    visited: set[str] = set()
    cluster_index = 0
    for name in sorted(graph):
        if name in visited:
            continue
        pending = [name]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(graph[current] - component)
        visited.update(component)
        output[f"cluster_{cluster_index:03d}"] = tuple(sorted(component))
        cluster_index += 1
    return output
