"""Multi-Objective Pareto Frontier, Non-Dominated Sorting, and Exact Hypervolume Engine.

Evaluates multi-agent cross-verification and routing topologies across three competing
systems objectives:
    1. Task Accuracy (maximize, in [0, 1])
    2. Mean Token Expenditure per Task (minimize, tokens)
    3. End-to-End Tail Latency P95 (minimize, milliseconds)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ConfigPoint:
    """Multi-objective evaluation summary of a multi-agent topology configuration."""

    config_id: str
    label: str
    topology_family: str
    accuracy: float
    mean_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float
    accuracy_std: float = 0.0
    easy_accuracy: float = 0.0
    medium_accuracy: float = 0.0
    hard_accuracy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def dominates_2d(a: ConfigPoint, b: ConfigPoint, tol: float = 1e-12) -> bool:
    """Return True iff ``a`` Pareto-dominates ``b`` on (Accuracy up, Token Cost down)."""
    not_worse = (a.accuracy >= b.accuracy - tol) and (a.mean_tokens <= b.mean_tokens + tol)
    strictly_better = (a.accuracy > b.accuracy + tol) or (a.mean_tokens < b.mean_tokens - tol)
    return bool(not_worse and strictly_better)


def dominates_3d(a: ConfigPoint, b: ConfigPoint, tol: float = 1e-12) -> bool:
    """Return True iff ``a`` Pareto-dominates ``b`` on (Accuracy up, Cost down, P95 Latency down)."""
    not_worse = (
        (a.accuracy >= b.accuracy - tol)
        and (a.mean_tokens <= b.mean_tokens + tol)
        and (a.p95_latency_ms <= b.p95_latency_ms + tol)
    )
    strictly_better = (
        (a.accuracy > b.accuracy + tol)
        or (a.mean_tokens < b.mean_tokens - tol)
        or (a.p95_latency_ms < b.p95_latency_ms - tol)
    )
    return bool(not_worse and strictly_better)


def pareto_frontier_2d(points: Sequence[ConfigPoint]) -> List[ConfigPoint]:
    """Extract the exact 2D Pareto frontier on (mean_tokens down, accuracy up), sorted by cost."""
    if not points:
        return []
    frontier: List[ConfigPoint] = []
    for i, cand in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i != j and dominates_2d(other, cand):
                dominated = True
                break
        if not dominated:
            # Deduplicate identical objective tuples consistently
            if not any(
                abs(f.accuracy - cand.accuracy) <= 1e-12
                and abs(f.mean_tokens - cand.mean_tokens) <= 1e-12
                for f in frontier
            ):
                frontier.append(cand)
    frontier.sort(key=lambda p: (p.mean_tokens, -p.accuracy))
    return frontier


def pareto_frontier_3d(points: Sequence[ConfigPoint]) -> List[ConfigPoint]:
    """Extract the exact 3D non-dominated frontier on (Accuracy up, Cost down, P95 Latency down)."""
    if not points:
        return []
    ranks, _ = fast_non_dominated_sort_3d(points)
    return ranks[0] if ranks else []


def fast_non_dominated_sort_3d(
    points: Sequence[ConfigPoint],
) -> Tuple[List[List[ConfigPoint]], Dict[str, int]]:
    """NSGA-II non-dominated sorting across (Accuracy up, Token Cost down, P95 Latency down).

    Returns:
        Tuple of (list of Pareto fronts from rank 1..K, dict mapping config_id -> 1-based rank).
    """
    n = len(points)
    if n == 0:
        return [], {}

    dominated_sets: List[List[int]] = [[] for _ in range(n)]
    dom_counts = np.zeros(n, dtype=np.int64)
    front_indices: List[List[int]] = [[]]

    for p_idx in range(n):
        for q_idx in range(n):
            if p_idx == q_idx:
                continue
            if dominates_3d(points[p_idx], points[q_idx]):
                dominated_sets[p_idx].append(q_idx)
            elif dominates_3d(points[q_idx], points[p_idx]):
                dom_counts[p_idx] += 1
        if dom_counts[p_idx] == 0:
            front_indices[0].append(p_idx)

    curr = 0
    while curr < len(front_indices) and front_indices[curr]:
        next_front: List[int] = []
        for p_idx in front_indices[curr]:
            for q_idx in dominated_sets[p_idx]:
                dom_counts[q_idx] -= 1
                if dom_counts[q_idx] == 0:
                    next_front.append(q_idx)
        if next_front:
            front_indices.append(next_front)
        curr += 1

    ranked_fronts: List[List[ConfigPoint]] = []
    rank_map: Dict[str, int] = {}
    for r_zero, idxs in enumerate(front_indices):
        if not idxs:
            continue
        rank = r_zero + 1
        front_pts = sorted((points[i] for i in idxs), key=lambda x: (x.mean_tokens, -x.accuracy))
        ranked_fronts.append(front_pts)
        for pt in front_pts:
            rank_map[pt.config_id] = rank

    return ranked_fronts, rank_map


def _hv_2d_max_coords(coords_2d: np.ndarray) -> float:
    """Exact 2D hypervolume of maximization points ``coords_2d`` in R_{>=0}^2 wrt origin (0, 0)."""
    if coords_2d.size == 0:
        return 0.0
    valid = coords_2d[(coords_2d[:, 0] > 0.0) & (coords_2d[:, 1] > 0.0)]
    if valid.shape[0] == 0:
        return 0.0
    # Sort descending by first coordinate x; sweep running maximum of y
    order = np.argsort(-valid[:, 0], kind="mergesort")
    pts = valid[order]
    hv = 0.0
    max_y = 0.0
    for x, y in pts:
        if y > max_y:
            hv += float(x) * float(y - max_y)
            max_y = float(y)
    return float(hv)


def compute_hypervolume_2d(
    points: Sequence[ConfigPoint],
    ref_accuracy: float = 0.40,
    ref_tokens: float = 8000.0,
    normalize: bool = True,
) -> float:
    """Exact 2D Hypervolume wrt worst-case reference ``(ref_accuracy, ref_tokens)``.

    Transforms ``(accuracy up, mean_tokens down)`` into positive attainment coordinates.
    """
    if not points:
        return 0.0
    acc_span = max(1.0 - ref_accuracy, 1e-12)
    tok_span = max(ref_tokens, 1e-12)

    coords = []
    for p in points:
        dx = p.accuracy - ref_accuracy
        dy = ref_tokens - p.mean_tokens
        if normalize:
            dx /= acc_span
            dy /= tok_span
        if dx > 0.0 and dy > 0.0:
            coords.append((dx, dy))
    if not coords:
        return 0.0
    return _hv_2d_max_coords(np.asarray(coords, dtype=np.float64))


def compute_hypervolume_3d(
    points: Sequence[ConfigPoint],
    ref_accuracy: float = 0.40,
    ref_tokens: float = 8000.0,
    ref_latency_ms: float = 6500.0,
    normalize: bool = True,
) -> float:
    """Exact 3D Hypervolume via z-slice sweep + exact 2D sweep-line cross-sections."""
    if not points:
        return 0.0
    acc_span = max(1.0 - ref_accuracy, 1e-12)
    tok_span = max(ref_tokens, 1e-12)
    lat_span = max(ref_latency_ms, 1e-12)

    coords = []
    for p in points:
        dx = p.accuracy - ref_accuracy
        dy = ref_tokens - p.mean_tokens
        dz = ref_latency_ms - p.p95_latency_ms
        if normalize:
            dx /= acc_span
            dy /= tok_span
            dz /= lat_span
        if dx > 0.0 and dy > 0.0 and dz > 0.0:
            coords.append((dx, dy, dz))

    if not coords:
        return 0.0

    arr = np.asarray(coords, dtype=np.float64)
    # Sort descending by z (3rd coordinate)
    order = np.argsort(-arr[:, 2], kind="mergesort")
    pts = arr[order]

    hv_3d = 0.0
    n = pts.shape[0]
    for k in range(n):
        z_curr = float(pts[k, 2])
        z_next = float(pts[k + 1, 2]) if (k + 1 < n) else 0.0
        dz = z_curr - z_next
        if dz > 1e-15:
            area_xy = _hv_2d_max_coords(pts[: k + 1, :2])
            hv_3d += dz * area_xy
    return float(hv_3d)


def compute_marginal_token_utility(frontier_2d: Sequence[ConfigPoint]) -> List[Dict[str, Any]]:
    """Compute marginal accuracy gain per +1,000 tokens along the sorted 2D Pareto frontier."""
    sorted_front = sorted(frontier_2d, key=lambda p: (p.mean_tokens, -p.accuracy))
    results: List[Dict[str, Any]] = []
    for idx, pt in enumerate(sorted_front):
        if idx == 0:
            results.append(
                {
                    "config_id": pt.config_id,
                    "label": pt.label,
                    "accuracy": pt.accuracy,
                    "mean_tokens": pt.mean_tokens,
                    "delta_accuracy_pp": 0.0,
                    "delta_tokens": 0.0,
                    "marginal_pp_per_1k_tokens": None,
                }
            )
        else:
            prev = sorted_front[idx - 1]
            d_acc_pp = (pt.accuracy - prev.accuracy) * 100.0
            d_tok = max(pt.mean_tokens - prev.mean_tokens, 1e-9)
            mtu = d_acc_pp / (d_tok / 1000.0)
            results.append(
                {
                    "config_id": pt.config_id,
                    "label": pt.label,
                    "accuracy": pt.accuracy,
                    "mean_tokens": pt.mean_tokens,
                    "delta_accuracy_pp": float(d_acc_pp),
                    "delta_tokens": float(d_tok),
                    "marginal_pp_per_1k_tokens": float(mtu),
                }
            )
    return results


def find_knee_point_2d(points: Sequence[ConfigPoint]) -> ConfigPoint:
    """Select the Pareto knee point maximizing perpendicular uplift above the extreme chord."""
    front = pareto_frontier_2d(points)
    if not front:
        raise ValueError("Cannot find knee point of empty configuration sequence.")
    if len(front) <= 2:
        return front[-1]

    min_tok = front[0].mean_tokens
    max_tok = front[-1].mean_tokens
    min_acc = front[0].accuracy
    max_acc = front[-1].accuracy

    tok_range = max(max_tok - min_tok, 1e-12)
    acc_range = max(max_acc - min_acc, 1e-12)

    best_pt = front[0]
    best_score = -float("inf")
    for pt in front:
        norm_cost = (pt.mean_tokens - min_tok) / tok_range
        norm_acc = (pt.accuracy - min_acc) / acc_range
        # Perpendicular signed distance (up to 1/sqrt(2)) above chord y = x
        chord_uplift = (norm_acc - norm_cost) / np.sqrt(2.0)
        if chord_uplift > best_score:
            best_score = float(chord_uplift)
            best_pt = pt
    return best_pt


def select_under_budget(
    points: Sequence[ConfigPoint],
    max_tokens: float,
    max_p95_latency_ms: Optional[float] = None,
) -> Optional[ConfigPoint]:
    """Return highest-accuracy configuration satisfying token budget and optional P95 SLA."""
    feasible = [
        p
        for p in points
        if p.mean_tokens <= max_tokens
        and (max_p95_latency_ms is None or p.p95_latency_ms <= max_p95_latency_ms)
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda p: (-p.accuracy, p.mean_tokens, p.p95_latency_ms))
    return feasible[0]
