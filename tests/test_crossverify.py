"""Comprehensive pytest suite for CrossVerify-Bench theory, tasks, Pareto engine, and topologies."""

from __future__ import annotations

import numpy as np
import pytest

from crossverify.markov_theory import (
    MarkovVerifierChain,
    VerificationParams,
    compute_phase_diagram_grid,
)
from crossverify.pareto import (
    ConfigPoint,
    compute_hypervolume_2d,
    compute_hypervolume_3d,
    compute_marginal_token_utility,
    dominates_2d,
    dominates_3d,
    fast_non_dominated_sort_3d,
    find_knee_point_2d,
    pareto_frontier_2d,
    select_under_budget,
)
from crossverify.tasks import (
    build_benchmark_suite,
    extract_prompt_features,
    summarize_benchmark_suite,
)
from crossverify.topologies import (
    AdversarialDebateDAG,
    CalibratedReplayBackend,
    GeneratorVerifierLoop,
    ParetoAdaptiveRouter,
    SelfConsistencyVote,
    SinglePass,
    build_benchmark_configurations,
)


# ---------------------------------------------------------------------------
# 1. Markov-Chain Verification Theory Tests
# ---------------------------------------------------------------------------


def test_markov_parameter_validation() -> None:
    with pytest.raises(ValueError):
        VerificationParams(p0=1.2, alpha=0.1, beta=0.2, q_i2c=0.5, q_c2i=0.3)
    with pytest.raises(ValueError):
        VerificationParams(p0=0.8, alpha=-0.05, beta=0.2, q_i2c=0.5, q_c2i=0.3)
    with pytest.raises(ValueError):
        VerificationParams(p0=0.8, alpha=0.1, beta=0.2, q_i2c=0.5, q_c2i=0.3, cost_ver=-10.0)


def test_verification_harm_theorem_and_critical_thresholds() -> None:
    # Strong generator (p0 = 0.86) paired with noisy verifier (alpha = 0.30, beta = 0.42)
    harm_params = VerificationParams(
        p0=0.86, alpha=0.30, beta=0.42, q_i2c=0.50, q_c2i=0.40
    )
    chain = MarkovVerifierChain(harm_params)
    assert not chain.is_net_positive_one_step()
    assert chain.one_step_gain() < -0.05
    p1 = chain.accuracy_at_step(1, early_stop=True)
    p3_homog = chain.accuracy_at_step(3, early_stop=False)
    assert p1 < harm_params.p0
    assert p3_homog < p1 < harm_params.p0

    # At exact critical alpha*(beta), single-step gain Delta p_1 must vanish to machine precision
    alpha_star = chain.critical_alpha_unclipped()
    assert 0.0 < alpha_star < 1.0
    boundary_chain = MarkovVerifierChain(
        VerificationParams(
            p0=harm_params.p0,
            alpha=alpha_star,
            beta=harm_params.beta,
            q_i2c=harm_params.q_i2c,
            q_c2i=harm_params.q_c2i,
        )
    )
    assert abs(boundary_chain.one_step_gain()) < 1e-14


@pytest.mark.parametrize("early_stop", [True, False])
def test_markov_closed_form_matches_monte_carlo_and_asymptotics(early_stop: bool) -> None:
    params = VerificationParams(
        p0=0.72,
        alpha=0.08,
        beta=0.22,
        q_i2c=0.64,
        q_c2i=0.28,
        cost_gen=700.0,
        cost_ver=450.0,
        cost_rep=750.0,
    )
    chain = MarkovVerifierChain(params)

    # T = 0 identity
    assert chain.accuracy_at_step(0, early_stop=early_stop) == pytest.approx(0.72)
    assert chain.expected_cost(0, early_stop=early_stop) == pytest.approx(700.0)

    # T = 1 must be identical between early_stop=True and early_stop=False
    assert chain.accuracy_at_step(1, early_stop=True) == pytest.approx(
        chain.accuracy_at_step(1, early_stop=False), abs=1e-14
    )
    assert chain.expected_cost(1, early_stop=True) == pytest.approx(
        chain.expected_cost(1, early_stop=False), abs=1e-12
    )

    # Monte Carlo agreement (< 0.4% accuracy error, < 1% relative cost error)
    for t_step in (1, 2, 4):
        mc = chain.simulate_monte_carlo(
            T=t_step, n_trials=60000, early_stop=early_stop, seed=42 + t_step
        )
        assert mc["abs_accuracy_error"] < 0.004
        assert mc["rel_cost_error"] < 0.01

    # Asymptotic convergence T -> 60 matches exact stationary_accuracy formula
    p_large = chain.accuracy_at_step(60, early_stop=early_stop)
    p_inf = chain.stationary_accuracy(early_stop=early_stop)
    assert p_large == pytest.approx(p_inf, abs=1e-10)


def test_phase_diagram_grid_shapes_and_signs() -> None:
    alphas = np.linspace(0.0, 0.5, 11)
    betas = np.linspace(0.0, 0.6, 13)
    AA, BB, dp = compute_phase_diagram_grid(0.80, 0.60, 0.35, alphas, betas, T=1)
    assert AA.shape == (13, 11)
    assert BB.shape == (13, 11)
    assert dp.shape == (13, 11)
    # Perfect verifier corner (alpha=0, beta=0) strictly improves accuracy
    assert dp[0, 0] > 0.0
    # High false-rejection + high false-acceptance corner degrades accuracy
    assert dp[-1, -1] < 0.0


# ---------------------------------------------------------------------------
# 2. Multi-Objective Pareto & Exact Hypervolume Tests
# ---------------------------------------------------------------------------


def test_pareto_dominance_and_exact_hypervolume_monotonicity() -> None:
    p_fast = ConfigPoint("c1", "Fast", "S1", accuracy=0.60, mean_tokens=500.0, p50_latency_ms=250.0, p95_latency_ms=400.0)
    p_mid = ConfigPoint("c2", "Mid", "S3", accuracy=0.80, mean_tokens=2000.0, p50_latency_ms=900.0, p95_latency_ms=1500.0)
    p_dom = ConfigPoint("c3", "Dominated", "S2", accuracy=0.75, mean_tokens=2500.0, p50_latency_ms=1100.0, p95_latency_ms=1800.0)

    assert dominates_2d(p_mid, p_dom)
    assert dominates_3d(p_mid, p_dom)
    assert not dominates_2d(p_fast, p_mid)
    assert not dominates_2d(p_mid, p_fast)

    front = pareto_frontier_2d([p_fast, p_mid, p_dom])
    assert [x.config_id for x in front] == ["c1", "c2"]

    # Adding a dominated point MUST leave 2D and 3D Hypervolume identically invariant
    hv2_base = compute_hypervolume_2d([p_fast, p_mid], ref_accuracy=0.40, ref_tokens=4000.0, normalize=False)
    hv2_with_dom = compute_hypervolume_2d([p_fast, p_mid, p_dom], ref_accuracy=0.40, ref_tokens=4000.0, normalize=False)
    assert hv2_base == pytest.approx(hv2_with_dom, abs=1e-12)

    # Exact manual 2D area check wrt (0.40, 4000):
    # p_mid: dx=0.40, dy=2000 -> 800; p_fast adds (3500 - 2000)=1500 height over dx=(0.60-0.40)=0.20 -> +300 => 1100.0!
    assert hv2_base == pytest.approx(1100.0, abs=1e-9)

    hv3_base = compute_hypervolume_3d(
        [p_fast, p_mid], ref_accuracy=0.40, ref_tokens=4000.0, ref_latency_ms=3000.0, normalize=True
    )
    hv3_with_dom = compute_hypervolume_3d(
        [p_fast, p_mid, p_dom], ref_accuracy=0.40, ref_tokens=4000.0, ref_latency_ms=3000.0, normalize=True
    )
    assert hv3_base == pytest.approx(hv3_with_dom, abs=1e-12)

    # Adding a strictly dominating adaptive router MUST strictly increase both HV_2D and HV_3D
    p_router = ConfigPoint("c4", "Router", "S5", accuracy=0.88, mean_tokens=1600.0, p50_latency_ms=700.0, p95_latency_ms=1200.0)
    hv3_improved = compute_hypervolume_3d(
        [p_fast, p_mid, p_dom, p_router], ref_accuracy=0.40, ref_tokens=4000.0, ref_latency_ms=3000.0, normalize=True
    )
    assert hv3_improved > hv3_base + 0.05

    ranks, rmap = fast_non_dominated_sort_3d([p_fast, p_mid, p_dom, p_router])
    assert rmap["c1"] == 1 and rmap["c4"] == 1
    assert rmap["c2"] == 2
    assert rmap["c3"] == 3


def test_knee_point_and_budget_selection() -> None:
    pts = [
        ConfigPoint("a", "A", "S1", 0.55, 400.0, 200.0, 350.0),
        ConfigPoint("b", "B", "S5", 0.89, 1800.0, 750.0, 1300.0),  # clear knee
        ConfigPoint("c", "C", "S4", 0.91, 6500.0, 2200.0, 3900.0),
    ]
    knee = find_knee_point_2d(pts)
    assert knee.config_id == "b"

    mtu = compute_marginal_token_utility(pts)
    assert len(mtu) == 3
    assert mtu[1]["marginal_pp_per_1k_tokens"] > mtu[2]["marginal_pp_per_1k_tokens"]

    assert select_under_budget(pts, max_tokens=2000.0).config_id == "b"
    assert select_under_budget(pts, max_tokens=2000.0, max_p95_latency_ms=1000.0).config_id == "a"
    assert select_under_budget(pts, max_tokens=200.0) is None


# ---------------------------------------------------------------------------
# 3. Verifiable 180-Task Benchmark & Zero-Oracle Feature Tests
# ---------------------------------------------------------------------------


def test_benchmark_suite_completeness_and_evaluators() -> None:
    suite = build_benchmark_suite()
    summary = summarize_benchmark_suite(suite)
    assert summary["total_tasks"] == 180
    assert summary["by_domain"] == {
        "code_patch": 60,
        "scientific_symbolic": 60,
        "evidence_crosscheck": 60,
    }
    assert summary["by_difficulty"] == {"easy": 60, "medium": 60, "hard": 60}

    # Observable prompt complexity strictly increases across difficulty strata
    comp = summary["mean_prompt_complexity_by_difficulty"]
    assert comp["easy"] < comp["medium"] < comp["hard"]

    for task in suite:
        assert task.evaluate(task.ground_truth) is True
        ok_gt, _ = task.runtime_microcheck(task.ground_truth)
        assert ok_gt is True
        for idx, dist in enumerate(task.distractors):
            assert task.evaluate(dist) is False
            ok_d, _ = task.runtime_microcheck(dist)
            if idx < 2:
                assert ok_d is False
            else:
                # Subtle semantic distractor passes runtime microcheck (proving zero oracle leakage)
                assert ok_d is True


# ---------------------------------------------------------------------------
# 4. Multi-Agent Topologies & Adaptive Router Tests
# ---------------------------------------------------------------------------


def test_all_five_topology_families_and_dag_critical_path() -> None:
    suite = build_benchmark_suite()
    sample_tasks = [suite[0], suite[70], suite[170]]
    backend = CalibratedReplayBackend()
    rng = np.random.default_rng(123)

    topologies = [
        SinglePass("t_s1", "S1", "fast-8b"),
        SelfConsistencyVote("t_s2", "S2", "mid-32b", k_samples=5),
        GeneratorVerifierLoop("t_s3", "S3", "frontier-70b", "tool-augmented-verifier", max_rounds=2),
        AdversarialDebateDAG("t_s4", "S4", "mid-32b", "frontier-70b", "frontier-70b", use_tool_in_judge=True),
        ParetoAdaptiveRouter("t_s5", "S5"),
    ]

    for topo in topologies:
        for task in sample_tasks:
            tr = topo.run_task(task, backend, rng)
            assert tr.total_tokens > 50.0
            assert tr.latency_ms > 10.0
            assert isinstance(tr.is_correct, bool)
            assert tr.is_correct == task.evaluate(tr.candidate)

    # Verify networkx DAG critical path calculation
    _, crit = AdversarialDebateDAG.build_execution_dag(
        {
            "source": 0.0,
            "proposer_A": 500.0,
            "proposer_B": 900.0,
            "cross_exam_A": 400.0,
            "cross_exam_B": 650.0,
            "judge": 800.0,
        }
    )
    assert crit == pytest.approx(900.0 + 650.0 + 800.0)


def test_router_budget_hierarchy_and_config_registry() -> None:
    configs = build_benchmark_configurations()
    assert len(configs) >= 26

    suite = build_benchmark_suite()
    backend = CalibratedReplayBackend()
    routers = [c for c in configs if isinstance(c, ParetoAdaptiveRouter)]
    assert len(routers) == 4

    mean_toks = {}
    for r_cfg in routers:
        rng = np.random.default_rng(777)
        toks = [r_cfg.run_task(t, backend, rng).total_tokens for t in suite]
        mean_toks[r_cfg.config_id] = float(np.mean(toks))

    assert (
        mean_toks["S5_Router_Lite"]
        < mean_toks["S5_Router_Eco"]
        < mean_toks["S5_Router_Balanced"]
        < mean_toks["S5_Router_Strict"]
    )


def test_benchmark_artifacts_and_markov_half_percent_guarantee() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    json_path = root / "results" / "crossverify_pareto_benchmark.json"
    csv_path = root / "results" / "topology_comparison.csv"
    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["n_configurations"] == 28
    assert payload["markov_validation"]["passes_half_percent_bound"] is True
    assert payload["markov_validation"]["max_monte_carlo_abs_error"] < 0.005
    assert payload["markov_validation"]["max_benchmark_s3_abs_error"] < 0.005

    savings = payload["pareto_summary"]["router_vs_static_savings"]
    assert savings["S5_Router_Eco"]["token_savings_pct"] >= 50.0
    assert savings["S5_Router_Balanced"]["token_savings_pct"] >= 50.0

    expected_figs = [
        "pareto_frontier_cost_vs_accuracy.png",
        "verifier_phase_diagram.png",
        "difficulty_stratified_breakdown.png",
        "routing_allocation_and_latency.png",
    ]
    for fig_name in expected_figs:
        fig_path = root / "assets" / fig_name
        assert fig_path.exists() and fig_path.stat().st_size > 100_000

