"""Execute the complete CrossVerify-Bench 28-configuration Pareto & Markov benchmark.

Outputs:
    - ``results/crossverify_pareto_benchmark.json``: Complete benchmark, stratified breakdowns,
      Pareto frontiers, Hypervolumes, router tier allocations, and Markov validation tables.
    - ``results/topology_comparison.csv``: Tabular summary across all 28 configurations.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossverify.markov_theory import MarkovVerifierChain, VerificationParams
from crossverify.pareto import (
    ConfigPoint,
    compute_hypervolume_2d,
    compute_hypervolume_3d,
    compute_marginal_token_utility,
    fast_non_dominated_sort_3d,
    find_knee_point_2d,
    pareto_frontier_2d,
    select_under_budget,
)
from crossverify.tasks import build_benchmark_suite, summarize_benchmark_suite
from crossverify.topologies import (
    CalibratedReplayBackend,
    GeneratorVerifierLoop,
    build_benchmark_configurations,
)


def run_markov_theory_validation(
    suite: List[Any], backend: CalibratedReplayBackend, configs: List[Any], emp_by_id: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate closed-form Markov equations against both synthetic rollouts & 180-task benchmark."""
    canonical_regimes = [
        ("StrongGen_WeakVer_T1_Early", VerificationParams(0.83, 0.29, 0.43, 0.52, 0.41, 1160, 340, 1240), 1, True),
        ("StrongGen_WeakVer_T3_Homog", VerificationParams(0.83, 0.29, 0.43, 0.52, 0.41, 1160, 340, 1240), 3, False),
        ("MidGen_CalibVer_T2_Early", VerificationParams(0.73, 0.045, 0.18, 0.65, 0.25, 670, 740, 730), 2, True),
        ("Frontier_ToolVer_T1_Early", VerificationParams(0.83, 0.029, 0.065, 0.85, 0.19, 1160, 680, 1240), 1, True),
        ("Frontier_ToolVer_T3_Early", VerificationParams(0.83, 0.029, 0.065, 0.85, 0.19, 1160, 680, 1240), 3, True),
        ("WeakGen_MidVer_T5_Homog", VerificationParams(0.55, 0.12, 0.30, 0.58, 0.34, 360, 520, 410), 5, False),
    ]

    mc_checks: List[Dict[str, Any]] = []
    for idx, (name, params, t_rounds, early) in enumerate(canonical_regimes):
        chain = MarkovVerifierChain(params)
        res = chain.simulate_monte_carlo(T=t_rounds, n_trials=60000, early_stop=early, seed=1000 + idx)
        res["regime"] = name
        res["one_step_gain_pp"] = chain.one_step_gain() * 100.0
        res["stationary_accuracy"] = chain.stationary_accuracy(early_stop=early)
        mc_checks.append(res)

    # Compare all S3_GenVerifyLoop benchmark runs against stratum-exact Markov theory
    s3_benchmark_checks: List[Dict[str, Any]] = []
    for cfg in configs:
        if isinstance(cfg, GeneratorVerifierLoop):
            ana = cfg.analytical_predictions(suite, backend)
            emp_acc = emp_by_id[cfg.config_id]["accuracy"]
            abs_err = abs(emp_acc - ana["overall_analytical_accuracy"])
            s3_benchmark_checks.append(
                {
                    "config_id": cfg.config_id,
                    "label": cfg.label,
                    "analytical_accuracy": ana["overall_analytical_accuracy"],
                    "empirical_accuracy": emp_acc,
                    "abs_accuracy_error": abs_err,
                    "by_difficulty_analytical": ana["by_difficulty_accuracy"],
                }
            )

    max_mc_err = max(c["abs_accuracy_error"] for c in mc_checks)
    max_s3_err = max(c["abs_accuracy_error"] for c in s3_benchmark_checks)
    return {
        "max_monte_carlo_abs_error": float(max_mc_err),
        "max_benchmark_s3_abs_error": float(max_s3_err),
        "passes_half_percent_bound": bool(max_mc_err < 0.005 and max_s3_err < 0.005),
        "canonical_regime_rollouts": mc_checks,
        "s3_benchmark_validation": s3_benchmark_checks,
    }


def main(n_seeds: int = 100) -> Dict[str, Any]:
    t_start = time.perf_counter()
    suite = build_benchmark_suite()
    suite_summary = summarize_benchmark_suite(suite)
    backend = CalibratedReplayBackend()
    configs = build_benchmark_configurations()

    raw_records: Dict[str, Dict[str, Any]] = {}
    config_points: List[ConfigPoint] = []

    for cfg_idx, cfg in enumerate(configs):
        seed_accs: List[float] = []
        all_tokens: List[float] = []
        all_lats: List[float] = []
        all_llm_calls: List[int] = []
        all_ver_calls: List[int] = []
        all_rep_calls: List[int] = []
        all_depths: List[int] = []

        diff_correct: Dict[str, List[int]] = defaultdict(list)
        diff_tokens: Dict[str, List[float]] = defaultdict(list)
        dom_correct: Dict[str, List[int]] = defaultdict(list)
        dom_tokens: Dict[str, List[float]] = defaultdict(list)
        router_tiers_by_diff: Dict[str, Counter[str]] = defaultdict(Counter)
        shortcircuit_hits = 0
        tier3_total = 0

        for s in range(n_seeds):
            rng = np.random.default_rng(202600 + cfg_idx * 97 + s)
            correct_this_seed = 0
            for task in suite:
                tr = cfg.run_task(task, backend, rng)
                c_int = int(tr.is_correct)
                correct_this_seed += c_int
                all_tokens.append(tr.total_tokens)
                all_lats.append(tr.latency_ms)
                all_llm_calls.append(tr.llm_calls)
                all_ver_calls.append(tr.verifier_calls)
                all_rep_calls.append(tr.repair_calls)
                all_depths.append(tr.dag_depth)

                diff_correct[task.difficulty].append(c_int)
                diff_tokens[task.difficulty].append(tr.total_tokens)
                dom_correct[task.domain].append(c_int)
                dom_tokens[task.domain].append(tr.total_tokens)

                if tr.router_tier is not None:
                    router_tiers_by_diff[task.difficulty][tr.router_tier] += 1
                    if tr.router_tier == "tier3_hetero_debate":
                        tier3_total += 1
                        if tr.metadata.get("disagreement_gate_shortcircuit"):
                            shortcircuit_hits += 1

            seed_accs.append(correct_this_seed / len(suite))

        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs, ddof=1)) if n_seeds > 1 else 0.0
        mean_tok = float(np.mean(all_tokens))
        p50_lat = float(np.percentile(all_lats, 50))
        p95_lat = float(np.percentile(all_lats, 95))

        easy_acc = float(np.mean(diff_correct["easy"]))
        med_acc = float(np.mean(diff_correct["medium"]))
        hard_acc = float(np.mean(diff_correct["hard"]))

        tier_alloc_pct: Dict[str, Dict[str, float]] = {}
        if router_tiers_by_diff:
            for d_key in ("easy", "medium", "hard"):
                cnts = router_tiers_by_diff[d_key]
                tot = max(sum(cnts.values()), 1)
                tier_alloc_pct[d_key] = {
                    t_name: float(cnts.get(t_name, 0) / tot)
                    for t_name in ("tier1_fast", "tier2_verified", "tier3_hetero_debate")
                }

        meta = {
            "mean_llm_calls": float(np.mean(all_llm_calls)),
            "mean_verifier_calls": float(np.mean(all_ver_calls)),
            "mean_repair_calls": float(np.mean(all_rep_calls)),
            "mean_dag_depth": float(np.mean(all_depths)),
            "tokens_by_difficulty": {
                k: float(np.mean(v)) for k, v in diff_tokens.items()
            },
            "accuracy_by_domain": {
                k: float(np.mean(v)) for k, v in dom_correct.items()
            },
            "tokens_by_domain": {
                k: float(np.mean(v)) for k, v in dom_tokens.items()
            },
            "router_tier_allocation": tier_alloc_pct,
            "tier3_shortcircuit_rate": float(shortcircuit_hits / tier3_total) if tier3_total > 0 else 0.0,
        }

        pt = ConfigPoint(
            config_id=cfg.config_id,
            label=cfg.label,
            topology_family=cfg.topology_family,
            accuracy=mean_acc,
            mean_tokens=mean_tok,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            accuracy_std=std_acc,
            easy_accuracy=easy_acc,
            medium_accuracy=med_acc,
            hard_accuracy=hard_acc,
            metadata=meta,
        )
        config_points.append(pt)
        raw_records[cfg.config_id] = pt.to_dict()

    # Pareto analysis: Static-only (S1-S4) vs Global (S1-S5)
    static_points = [p for p in config_points if not p.topology_family.startswith("S5_")]
    front_2d_all = pareto_frontier_2d(config_points)
    front_2d_static = pareto_frontier_2d(static_points)
    front_2d_ids = {p.config_id for p in front_2d_all}
    front_2d_static_ids = {p.config_id for p in front_2d_static}

    _, rank_map_3d = fast_non_dominated_sort_3d(config_points)
    hv2_static = compute_hypervolume_2d(static_points, ref_accuracy=0.50, ref_tokens=9500.0)
    hv2_all = compute_hypervolume_2d(config_points, ref_accuracy=0.50, ref_tokens=9500.0)
    hv3_static = compute_hypervolume_3d(
        static_points, ref_accuracy=0.50, ref_tokens=9500.0, ref_latency_ms=8000.0
    )
    hv3_all = compute_hypervolume_3d(
        config_points, ref_accuracy=0.50, ref_tokens=9500.0, ref_latency_ms=8000.0
    )

    knee_pt = find_knee_point_2d(config_points)
    mtu_table = compute_marginal_token_utility(front_2d_all)

    # Compute exact token savings of ParetoAdaptiveRouter vs static topologies
    router_savings: Dict[str, Any] = {}
    for r_id in ("S5_Router_Eco", "S5_Router_Balanced", "S5_Router_Strict"):
        r_pt = next(p for p in config_points if p.config_id == r_id)
        # Cheapest static config achieving >= r_pt.accuracy - 0.004 (matching accuracy band)
        matching_static = [
            sp for sp in static_points if sp.accuracy >= r_pt.accuracy - 0.004
        ]
        if matching_static:
            best_static = min(matching_static, key=lambda x: x.mean_tokens)
        else:
            best_static = max(static_points, key=lambda x: x.accuracy)
        savings_pct = (1.0 - r_pt.mean_tokens / best_static.mean_tokens) * 100.0
        lat_savings_pct = (1.0 - r_pt.p95_latency_ms / best_static.p95_latency_ms) * 100.0
        router_savings[r_id] = {
            "router_label": r_pt.label,
            "router_accuracy": r_pt.accuracy,
            "router_tokens": r_pt.mean_tokens,
            "router_p95_ms": r_pt.p95_latency_ms,
            "matched_static_id": best_static.config_id,
            "matched_static_label": best_static.label,
            "matched_static_accuracy": best_static.accuracy,
            "matched_static_tokens": best_static.mean_tokens,
            "matched_static_p95_ms": best_static.p95_latency_ms,
            "token_savings_pct": float(savings_pct),
            "p95_latency_reduction_pct": float(lat_savings_pct),
        }

    # Enrich records with Pareto flags and save CSV + JSON
    csv_rows: List[Dict[str, Any]] = []
    for p in config_points:
        rec = raw_records[p.config_id]
        rec["is_pareto_2d"] = p.config_id in front_2d_ids
        rec["is_static_pareto_2d"] = p.config_id in front_2d_static_ids
        rec["pareto_rank_3d"] = int(rank_map_3d[p.config_id])
        rec["is_pareto_3d"] = int(rank_map_3d[p.config_id]) == 1

        csv_rows.append(
            {
                "config_id": p.config_id,
                "label": p.label,
                "topology_family": p.topology_family,
                "accuracy": round(p.accuracy, 5),
                "accuracy_std": round(p.accuracy_std, 5),
                "easy_accuracy": round(p.easy_accuracy, 5),
                "medium_accuracy": round(p.medium_accuracy, 5),
                "hard_accuracy": round(p.hard_accuracy, 5),
                "code_patch_acc": round(p.metadata["accuracy_by_domain"]["code_patch"], 5),
                "scientific_symbolic_acc": round(
                    p.metadata["accuracy_by_domain"]["scientific_symbolic"], 5
                ),
                "evidence_crosscheck_acc": round(
                    p.metadata["accuracy_by_domain"]["evidence_crosscheck"], 5
                ),
                "mean_tokens": round(p.mean_tokens, 2),
                "p50_latency_ms": round(p.p50_latency_ms, 2),
                "p95_latency_ms": round(p.p95_latency_ms, 2),
                "mean_llm_calls": round(p.metadata["mean_llm_calls"], 3),
                "is_pareto_2d": rec["is_pareto_2d"],
                "pareto_rank_3d": rec["pareto_rank_3d"],
            }
        )

    markov_val = run_markov_theory_validation(suite, backend, configs, raw_records)
    elapsed_s = time.perf_counter() - t_start

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "topology_comparison.csv"
    df.to_csv(csv_path, index=False)

    budget_recommendations = {
        f"budget_{b}_tok": (
            select_under_budget(config_points, max_tokens=float(b)).config_id
            if select_under_budget(config_points, max_tokens=float(b)) is not None
            else None
        )
        for b in (600, 1200, 2200, 3200, 5000)
    }

    summary_payload = {
        "benchmark_name": "CrossVerify-Bench",
        "version": "0.2.0",
        "n_configurations": len(config_points),
        "n_seeds": n_seeds,
        "elapsed_seconds": round(elapsed_s, 3),
        "suite_summary": suite_summary,
        "pareto_summary": {
            "global_2d_frontier_ids": [p.config_id for p in front_2d_all],
            "static_2d_frontier_ids": [p.config_id for p in front_2d_static],
            "knee_point_config_id": knee_pt.config_id,
            "knee_point_label": knee_pt.label,
            "hypervolume_2d_static": round(hv2_static, 6),
            "hypervolume_2d_global": round(hv2_all, 6),
            "hypervolume_2d_gain_pct": round((hv2_all / hv2_static - 1.0) * 100.0, 3),
            "hypervolume_3d_static": round(hv3_static, 6),
            "hypervolume_3d_global": round(hv3_all, 6),
            "hypervolume_3d_gain_pct": round((hv3_all / hv3_static - 1.0) * 100.0, 3),
            "marginal_token_utility": mtu_table,
            "budget_constrained_selections": budget_recommendations,
            "router_vs_static_savings": router_savings,
        },
        "markov_validation": markov_val,
        "configurations": [raw_records[p.config_id] for p in config_points],
    }

    json_path = out_dir / "crossverify_pareto_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"=== CrossVerify-Bench Completed in {elapsed_s:.2f}s ({len(config_points)} configs x 180 tasks x {n_seeds} seeds) ===")
    print(f"2D Hypervolume: Static={hv2_static:.4f} -> Global={hv2_all:.4f} (+{(hv2_all/hv2_static - 1)*100:.2f}%)")
    print(f"3D Hypervolume: Static={hv3_static:.4f} -> Global={hv3_all:.4f} (+{(hv3_all/hv3_static - 1)*100:.2f}%)")
    print(f"Knee Point: {knee_pt.config_id} ({knee_pt.label}) | Acc={knee_pt.accuracy*100:.2f}% | Tokens={knee_pt.mean_tokens:.0f}")
    print(
        f"Markov Validation Max Error: MC={markov_val['max_monte_carlo_abs_error']*100:.3f}%, "
        f"S3-Bench={markov_val['max_benchmark_s3_abs_error']*100:.3f}% "
        f"(Pass <0.5%: {markov_val['passes_half_percent_bound']})"
    )
    print("\n--- Router Token Savings vs Matching Static Topologies ---")
    for k, v in router_savings.items():
        print(
            f"  {v['router_label']}: Acc={v['router_accuracy']*100:.2f}% @ {v['router_tokens']:.0f} tok "
            f"vs {v['matched_static_label']} (Acc={v['matched_static_accuracy']*100:.2f}% @ {v['matched_static_tokens']:.0f} tok) "
            f"=> Token Savings = {v['token_savings_pct']:.1f}%, P95 Latency Reduction = {v['p95_latency_reduction_pct']:.1f}%"
        )
    return summary_payload


if __name__ == "__main__":
    main()
