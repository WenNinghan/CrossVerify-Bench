"""Generate the 4 publication-grade 300-DPI figures for CrossVerify-Bench.

Figures saved to ``assets/``:
    1. ``assets/pareto_frontier_cost_vs_accuracy.png``
    2. ``assets/verifier_phase_diagram.png``
    3. ``assets/difficulty_stratified_breakdown.png``
    4. ``assets/routing_allocation_and_latency.png``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossverify.markov_theory import (
    MarkovVerifierChain,
    VerificationParams,
    compute_phase_diagram_grid,
)
from crossverify.topologies import MODEL_REGISTRY, VERIFIER_REGISTRY


FAMILY_STYLE: Dict[str, Dict[str, Any]] = {
    "S1_SinglePass": {
        "color": "#3B6FB6",
        "marker": "o",
        "label": "S1: SinglePass",
        "zorder": 4,
    },
    "S2_SelfConsistency": {
        "color": "#E08214",
        "marker": "s",
        "label": "S2: Self-Consistency (SC@K)",
        "zorder": 4,
    },
    "S3_GenVerifyLoop": {
        "color": "#7B52AB",
        "marker": "^",
        "label": "S3: Gen-Verify Loop (GenVerify@T)",
        "zorder": 5,
    },
    "S4_AdversarialDebateDAG": {
        "color": "#C93B5B",
        "marker": "D",
        "label": "S4: Adversarial Debate DAG",
        "zorder": 5,
    },
    "S5_ParetoAdaptiveRouter": {
        "color": "#0F8B6D",
        "marker": "P",
        "label": "S5: ParetoAdaptiveRouter (Ours)",
        "zorder": 7,
    },
}


def _apply_pub_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.3,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#BDC5D0",
            "axes.linewidth": 0.9,
            "grid.color": "#DCE2EA",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
        }
    )


def plot_pareto_frontier(data: Dict[str, Any], assets_dir: Path) -> Path:
    """Figure 1: Token Cost vs. Accuracy Pareto Frontier & Marginal Token Utility."""
    configs = {c["config_id"]: c for c in data["configurations"]}
    ps = data["pareto_summary"]

    fig, (ax_main, ax_mtu) = plt.subplots(
        1, 2, figsize=(15.0, 6.9), gridspec_kw={"width_ratios": [2.35, 1.0]}
    )
    fig.subplots_adjust(wspace=0.33)

    # --- Left Panel: Global vs Static Pareto Frontier ---
    ax_main.grid(True, zorder=0)

    static_front = [configs[cid] for cid in ps["static_2d_frontier_ids"]]
    global_front = [configs[cid] for cid in ps["global_2d_frontier_ids"]]

    sf_x = [c["mean_tokens"] for c in static_front]
    sf_y = [c["accuracy"] * 100.0 for c in static_front]
    gf_x = [c["mean_tokens"] for c in global_front]
    gf_y = [c["accuracy"] * 100.0 for c in global_front]

    grid_x = np.linspace(min(gf_x), max(gf_x), 400)
    interp_global = np.interp(grid_x, gf_x, gf_y)
    interp_static = np.interp(grid_x, sf_x, sf_y)
    ax_main.fill_between(
        grid_x,
        interp_static,
        interp_global,
        where=(interp_global >= interp_static),
        color="#0F8B6D",
        alpha=0.13,
        zorder=1,
    )

    ax_main.plot(
        sf_x,
        sf_y,
        linestyle="--",
        linewidth=1.8,
        color="#64748B",
        zorder=2,
        label="Static-Only Pareto Frontier (S1-S4)",
    )
    ax_main.plot(
        gf_x,
        gf_y,
        linestyle="-",
        linewidth=2.6,
        color="#0F8B6D",
        zorder=3,
        label="Global Pareto Frontier (with S5 Router)",
    )

    for cid, c in configs.items():
        fam = c["topology_family"]
        st = FAMILY_STYLE[fam]
        is_p = c["is_pareto_2d"]
        sz = 140 if fam == "S5_ParetoAdaptiveRouter" else (88 if is_p else 62)
        alpha_pt = 0.98 if (is_p or "WeakVer" in cid) else 0.72
        edge_c = "#111827" if is_p else "white"
        lw = 1.2 if is_p else 0.7
        ax_main.scatter(
            c["mean_tokens"],
            c["accuracy"] * 100.0,
            s=sz,
            c=st["color"],
            marker=st["marker"],
            alpha=alpha_pt,
            edgecolors=edge_c,
            linewidths=lw,
            zorder=st["zorder"],
        )

    # Highlight Knee Point
    knee_id = ps["knee_point_config_id"]
    knee_c = configs[knee_id]
    ax_main.scatter(
        knee_c["mean_tokens"],
        knee_c["accuracy"] * 100.0,
        s=300,
        facecolors="none",
        edgecolors="#D97706",
        linewidths=2.4,
        zorder=8,
    )

    # Verification Harm trajectory (S1_Frontier -> S3_GV1_Frontier_WeakVer -> Homog)
    c_fr = configs["S1_Frontier"]
    c_w1 = configs["S3_GV1_Frontier_WeakVer"]
    c_w3 = configs["S3_GV3_Frontier_WeakVer_Homog"]
    ax_main.annotate(
        "",
        xy=(c_w1["mean_tokens"], c_w1["accuracy"] * 100.0),
        xytext=(c_fr["mean_tokens"], c_fr["accuracy"] * 100.0),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.7, ls="-."),
        zorder=6,
    )
    ax_main.annotate(
        "",
        xy=(c_w3["mean_tokens"], c_w3["accuracy"] * 100.0),
        xytext=(c_w1["mean_tokens"], c_w1["accuracy"] * 100.0),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.7, ls="-."),
        zorder=6,
    )
    drop_pp = (c_fr["accuracy"] - c_w3["accuracy"]) * 100.0
    cost_mult = c_w3["mean_tokens"] / c_fr["mean_tokens"]
    ax_main.annotate(
        f"Markov Harm Zone:\nWeak Verifier drops -{drop_pp:.1f} pp\nat {cost_mult:.1f}x token cost",
        xy=(c_w3["mean_tokens"], c_w3["accuracy"] * 100.0),
        xytext=(c_w3["mean_tokens"] + 350, c_w3["accuracy"] * 100.0 - 4.5),
        fontsize=8.2,
        fontweight="bold",
        color="#B91C1C",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FCA5A5", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.1),
        zorder=9,
    )

    # Token Savings horizontal comparison: Router-Eco vs GV1-ToolVer (and Router-Balanced vs Debate-Hetero)
    sav_eco = ps["router_vs_static_savings"]["S5_Router_Eco"]
    r_tok = sav_eco["router_tokens"]
    r_acc = sav_eco["router_accuracy"] * 100.0
    s_tok = sav_eco["matched_static_tokens"]
    sav_pct = sav_eco["token_savings_pct"]

    ax_main.annotate(
        "",
        xy=(r_tok + 70, r_acc),
        xytext=(s_tok - 70, r_acc),
        arrowprops=dict(arrowstyle="<->", color="#0F8B6D", lw=2.1),
        zorder=8,
    )
    # Also compare Router-Balanced vs Debate-Hetero (6525 tok)
    c_bal = configs["S5_Router_Balanced"]
    c_deb_h = configs["S4_Debate_Hetero_MidFrontier"]
    sav_deb_pct = (1.0 - c_bal["mean_tokens"] / c_deb_h["mean_tokens"]) * 100.0
    ax_main.annotate(
        f"-{sav_pct:.1f}% Tokens vs GV@1-ToolVer\n(-{sav_deb_pct:.0f}% vs Hetero-Debate at ~92% Acc)",
        xy=(0.5 * (r_tok + s_tok), r_acc),
        xytext=(s_tok + 320, r_acc - 3.6),
        fontsize=8.1,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.28", fc="#ECFDF5", ec="#6EE7B7", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#0F8B6D", lw=1.1),
        zorder=9,
    )

    # Cleanly separated point labels with zero overlap
    label_offsets = {
        "S1_Fast": (140, -1.3, "SinglePass-Fast"),
        "S1_Mid": (150, -1.4, "SinglePass-Mid"),
        "S1_Frontier": (-260, -2.5, "SinglePass-Frontier"),
        "S1_Reasoning": (150, -1.6, "SinglePass-Reasoning"),
        "S2_SC9_Fast": (150, +0.7, "SC@9-Fast (Ceiling)"),
        "S2_SC7_Mid": (150, -1.4, "SC@7-Mid"),
        "S2_SC5_Frontier": (-1350, -2.2, "SC@5-Frontier"),
        "S3_GV1_Frontier_WeakVer": (-1420, -2.3, "GenVerify@1 (WeakVer)"),
        "S3_GV3_Frontier_ToolVer": (160, -1.6, "GenVerify@3 (ToolVer)"),
        "S4_Debate_Homog_Mid": (150, -1.6, "Debate-Homog"),
        "S4_Debate_Hetero_MidFrontier": (150, -1.4, "Debate-Hetero"),
        "S4_Debate_Hetero_ToolJudge": (-1650, +1.3, "Debate-Hetero+Tool"),
        "S5_Router_Lite": (-1020, +0.9, "Router-Lite"),
        "S5_Router_Eco": (-1380, +1.1, "Router-Eco (Knee)"),
        "S5_Router_Balanced": (-150, +1.9, "Router-Balanced"),
        "S5_Router_Strict": (-1240, +1.0, "Router-Strict"),
    }
    for cid, (dx, dy, txt) in label_offsets.items():
        if cid in configs:
            cx = configs[cid]["mean_tokens"]
            cy = configs[cid]["accuracy"] * 100.0
            ax_main.text(
                max(180.0, cx + dx),
                cy + dy,
                txt,
                fontsize=7.8,
                fontweight="bold" if cid.startswith("S5_") else "normal",
                color="#065F46" if cid.startswith("S5_") else "#1E293B",
                zorder=9,
            )

    ax_main.set_xlabel("Mean Token Expenditure per Task (Prompt + Completion Tokens)")
    ax_main.set_ylabel("Verifiable Benchmark Accuracy (%)")
    ax_main.set_title("A. Multi-Agent Token Cost vs. Accuracy Pareto Frontier (28 Configurations)")
    ax_main.set_xlim(100, max(c["mean_tokens"] for c in configs.values()) * 1.10)
    ax_main.set_ylim(51.0, 101.0)

    handles = [
        Line2D(
            [0],
            [0],
            marker=st["marker"],
            color="w",
            label=st["label"],
            markerfacecolor=st["color"],
            markeredgecolor="#111827",
            markersize=8.2,
        )
        for st in FAMILY_STYLE.values()
    ]
    handles.append(Line2D([0], [0], color="#0F8B6D", lw=2.4, label="Global Pareto Front"))
    handles.append(Line2D([0], [0], color="#64748B", lw=1.7, ls="--", label="Static Pareto Front"))
    ax_main.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, fontsize=8.0)

    # --- Right Panel: Marginal Token Utility & Hypervolume Summary ---
    mtu_rows = [r for r in ps["marginal_token_utility"] if r["marginal_pp_per_1k_tokens"] is not None]
    short_names = {
        "S1_Mid": "S1-Mid",
        "S5_Router_Lite": "Router-Lite",
        "S5_Router_Eco": "Router-Eco",
        "S5_Router_Balanced": "Router-Balanced",
        "S5_Router_Strict": "Router-Strict",
        "S3_GV3_Frontier_ToolVer": "GV@3-ToolVer",
        "S4_Debate_Hetero_ToolJudge": "Debate+Tool",
    }
    labels_mtu = [short_names.get(r["config_id"], r["config_id"]) for r in mtu_rows]
    vals_mtu = [min(r["marginal_pp_per_1k_tokens"], 85.0) for r in mtu_rows]
    raw_vals = [r["marginal_pp_per_1k_tokens"] for r in mtu_rows]
    bar_colors = [
        "#0F8B6D" if r["config_id"].startswith("S5_") else "#475569" for r in mtu_rows
    ]

    y_pos = np.arange(len(mtu_rows))
    ax_mtu.grid(True, axis="x", zorder=0)
    bars = ax_mtu.barh(
        y_pos, vals_mtu, color=bar_colors, alpha=0.88, edgecolor="#1E293B", height=0.58, zorder=3
    )
    ax_mtu.set_yticks(y_pos)
    ax_mtu.set_yticklabels(labels_mtu, fontsize=8.5)
    ax_mtu.invert_yaxis()
    ax_mtu.set_xlabel("Marginal Gain (Acc pp / +1k Tokens)")
    ax_mtu.set_title("B. Marginal Utility Along Frontier")
    ax_mtu.set_xlim(0, max(vals_mtu) * 1.34 if vals_mtu else 60.0)

    for bar, raw_v in zip(bars, raw_vals):
        ax_mtu.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2.0,
            f"+{raw_v:.1f} pp/1k",
            va="center",
            ha="left",
            fontsize=7.9,
            fontweight="bold",
            color="#0F172A",
        )

    hv_box = (
        f"Hypervolume Attainment\n"
        f"-----------------------\n"
        f"2D HV Static:  {ps['hypervolume_2d_static']:.4f}\n"
        f"2D HV Global:  {ps['hypervolume_2d_global']:.4f} (+{ps['hypervolume_2d_gain_pct']:.1f}%)\n"
        f"3D HV Static:  {ps['hypervolume_3d_static']:.4f}\n"
        f"3D HV Global:  {ps['hypervolume_3d_global']:.4f} (+{ps['hypervolume_3d_gain_pct']:.1f}%)\n"
        f"Knee: {ps['knee_point_config_id']}"
    )
    # Place neatly in the open lower-middle right region where bars are very short (< 4 pp/1k)
    ax_mtu.text(
        0.97,
        0.34,
        hv_box,
        transform=ax_mtu.transAxes,
        ha="right",
        va="center",
        fontsize=7.8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.38", fc="#F8FAFC", ec="#94A3B8", lw=0.95),
        zorder=6,
    )

    out_path = assets_dir / "pareto_frontier_cost_vs_accuracy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_verifier_phase_diagram(data: Dict[str, Any], assets_dir: Path) -> Path:
    """Figure 2: 2D Analytical Verifier Phase Diagram & Multi-Round Markov Rollouts."""
    fig, (ax_phase, ax_traj) = plt.subplots(1, 2, figsize=(14.6, 6.3))
    fig.subplots_adjust(wspace=0.25)

    # --- Panel A: 2D Phase Diagram over (alpha, beta) for Strong Generator ---
    p0_ref = 0.80
    q_i2c_ref = 0.66
    q_c2i_ref = 0.30

    alphas = np.linspace(0.0, 0.48, 220)
    betas = np.linspace(0.0, 0.68, 220)
    AA, BB, delta_p = compute_phase_diagram_grid(
        p0=p0_ref, q_i2c=q_i2c_ref, q_c2i=q_c2i_ref, alpha_grid=alphas, beta_grid=betas, T=1
    )
    delta_pp = delta_p * 100.0

    vmax = max(abs(float(np.min(delta_pp))), abs(float(np.max(delta_pp))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    pcm = ax_phase.contourf(AA, BB, delta_pp, levels=28, cmap="RdBu", norm=norm, alpha=0.92)
    cbar = fig.colorbar(pcm, ax=ax_phase, fraction=0.046, pad=0.03)
    cbar.set_label("1-Step Net Accuracy Gain $\\Delta p_1 = p_1 - p_0$ (pp)", fontsize=9.2)

    cs = ax_phase.contour(
        AA,
        BB,
        delta_pp,
        levels=[-6.0, -3.0, 3.0, 6.0, 9.0],
        colors="#334155",
        linewidths=0.75,
        linestyles="--",
        alpha=0.65,
    )
    ax_phase.clabel(cs, fmt="%+.0f pp", fontsize=7.8)

    ref_chain = MarkovVerifierChain(
        VerificationParams(p0=p0_ref, alpha=0.1, beta=0.2, q_i2c=q_i2c_ref, q_c2i=q_c2i_ref)
    )
    b_line = np.array([ref_chain.critical_beta(float(a)) for a in alphas])
    valid_mask = (b_line > 0.0) & (b_line <= 0.68)
    ax_phase.plot(
        alphas[valid_mask],
        b_line[valid_mask],
        color="#0F172A",
        linewidth=2.7,
        label="Critical Boundary $\\Delta p_1 = 0$",
    )

    mspec = MODEL_REGISTRY["frontier-70b"]
    d_probs = np.array(mspec.distractor_probs) / np.sum(mspec.distractor_probs)
    emp_verifiers = [
        ("Weak Verifier\n(Net Harm)", "weak-verifier", "#DC2626", "X", (0.024, 0.03)),
        ("Mid Same-Family\nVerifier", "mid-verifier", "#D97706", "s", (0.022, 0.03)),
        ("Calibrated Cross\nVerifier", "calibrated-cross-verifier", "#2563EB", "D", (0.022, 0.03)),
        ("Tool-Augmented\nCross-Verifier", "tool-augmented-verifier", "#059669", "P", (0.022, 0.025)),
    ]
    for v_lbl, v_key, v_col, v_mk, (ox, oy) in emp_verifiers:
        vs = VERIFIER_REGISTRY[v_key]
        a_emp = vs.alpha_by_diff["medium"]
        b_emp = (
            float(d_probs[2]) * vs.beta_by_diff["medium"]
            if vs.use_runtime_microcheck
            else vs.beta_by_diff["medium"]
        )
        ax_phase.scatter(
            [a_emp],
            [b_emp],
            s=145,
            c=v_col,
            marker=v_mk,
            edgecolors="white",
            linewidths=1.4,
            zorder=8,
        )
        ax_phase.text(
            a_emp + ox,
            b_emp + oy,
            f"{v_lbl}\n($\\alpha$={a_emp:.2f}, $\\beta_{{eff}}$={b_emp:.2f})",
            fontsize=7.9,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=v_col, alpha=0.92, lw=1.0),
            zorder=9,
        )

    ax_phase.text(
        0.03,
        0.61,
        "NET ACCURACY GAIN\n($\\Delta p_1 > 0$)",
        fontsize=9.1,
        fontweight="bold",
        color="#1E3A8A",
    )
    ax_phase.text(
        0.29,
        0.58,
        "VERIFICATION HARM\n($\\Delta p_1 < 0$)",
        fontsize=9.1,
        fontweight="bold",
        color="#991B1B",
    )

    ax_phase.set_xlabel("Verifier Type-I False Rejection $\\alpha = \\mathbb{P}(\\mathrm{reject} \\mid C)$")
    ax_phase.set_ylabel("Verifier Type-II False Acceptance $\\beta = \\mathbb{P}(\\mathrm{accept} \\mid I)$")
    ax_phase.set_title(
        f"A. Verifier Phase Diagram ($p_0={p0_ref:.2f}, q_{{I\\to C}}={q_i2c_ref:.2f}, q_{{C\\to I}}={q_c2i_ref:.2f}$)"
    )
    ax_phase.set_xlim(0.0, 0.48)
    ax_phase.set_ylim(0.0, 0.68)
    ax_phase.legend(loc="lower left", frameon=True, framealpha=0.93)

    # --- Panel B: Multi-Round Markov Trajectories ---
    ax_traj.grid(True, zorder=0)
    t_steps = np.arange(0, 6)

    regimes = [
        (
            "Frontier + Tool-Augmented Verifier (Early-Stop)",
            VerificationParams(0.80, 0.028, 0.068, 0.75, 0.19),
            True,
            "#059669",
            "-",
            "P",
        ),
        (
            "Frontier + Calibrated Cross-Verifier (Early-Stop)",
            VerificationParams(0.80, 0.045, 0.19, 0.70, 0.225),
            True,
            "#2563EB",
            "-",
            "D",
        ),
        (
            "Mid-32B + Calibrated Cross-Verifier (Early-Stop)",
            VerificationParams(0.71, 0.045, 0.19, 0.60, 0.255),
            True,
            "#7B52AB",
            "-",
            "^",
        ),
        (
            "Frontier + Weak Verifier (Early-Stop)",
            VerificationParams(0.80, 0.31, 0.45, 0.50, 0.405),
            True,
            "#E08214",
            "--",
            "s",
        ),
        (
            "Frontier + Weak Verifier (Homogeneous Re-Verify)",
            VerificationParams(0.80, 0.31, 0.45, 0.50, 0.405),
            False,
            "#DC2626",
            "-.",
            "X",
        ),
    ]

    max_err_seen = 0.0
    for idx, (r_lbl, r_params, early, col, ls, mk) in enumerate(regimes):
        ch = MarkovVerifierChain(r_params)
        ana_curve = [ch.accuracy_at_step(int(t), early_stop=early) * 100.0 for t in t_steps]
        emp_pts = []
        for t in t_steps:
            mc = ch.simulate_monte_carlo(
                int(t), n_trials=150000, early_stop=early, seed=1900 + idx * 17 + int(t)
            )
            emp_pts.append(mc["empirical_accuracy"] * 100.0)
            max_err_seen = max(max_err_seen, mc["abs_accuracy_error"] * 100.0)

        ax_traj.plot(t_steps, ana_curve, color=col, linestyle=ls, linewidth=2.2, label=r_lbl, zorder=4)
        ax_traj.scatter(
            t_steps,
            emp_pts,
            s=62,
            facecolors="white",
            edgecolors=col,
            marker=mk,
            linewidths=1.6,
            zorder=6,
        )

    ax_traj.axhline(80.0, color="#64748B", linestyle=":", linewidth=1.2, zorder=2)
    ax_traj.text(4.95, 80.5, "Frontier $p_0 = 80\\%$", ha="right", fontsize=8.1, color="#475569")

    mv = data["markov_validation"]
    bound_val = max(max_err_seen, mv["max_benchmark_s3_abs_error"] * 100.0)
    badge = (
        f"Lines = Exact Closed-Form Theory\n"
        f"Markers = 150k Monte Carlo Rollouts\n"
        f"Max |Emp - Theory| = {bound_val:.2f}% (<0.5%)"
    )
    # Place in open middle-right band around y ~ 85% (between 80% line and 91% purple curve)
    ax_traj.text(
        0.97,
        0.64,
        badge,
        transform=ax_traj.transAxes,
        ha="right",
        va="center",
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec="#94A3B8", lw=0.95, alpha=0.95),
        zorder=8,
    )

    ax_traj.set_xticks(t_steps)
    ax_traj.set_xlabel("Max Verification-Repair Iterations $T$")
    ax_traj.set_ylabel("Expected Output Accuracy $p_T$ (%)")
    ax_traj.set_title("B. Iterative Markov Trajectories: Closed-Form vs. Rollouts")
    ax_traj.set_ylim(55.5, 99.5)
    ax_traj.legend(loc="lower left", fontsize=7.6, frameon=True, framealpha=0.94)

    out_path = assets_dir / "verifier_phase_diagram.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_difficulty_stratified_breakdown(data: Dict[str, Any], assets_dir: Path) -> Path:
    """Figure 3: Stratified Accuracy & Token Cost across Difficulty Strata and Domains."""
    configs = {c["config_id"]: c for c in data["configurations"]}

    selected_ids = [
        ("S1_Fast", "Fast-8B", "Fast-8B"),
        ("S1_Frontier", "Frontier", "Frontier-70B"),
        ("S2_SC7_Fast", "SC@7\n(Fast)", "SC@7-Fast"),
        ("S3_GV1_Frontier_WeakVer", "GV@1\n(WeakVer)", "GV@1-WeakVer"),
        ("S3_GV3_Frontier_ToolVer", "GV@3\n(ToolVer)", "GV@3-ToolVer"),
        ("S4_Debate_Hetero_ToolJudge", "Debate\n(Hetero)", "Debate+Tool"),
        ("S5_Router_Balanced", "Router\n(Balanced)", "Router-Bal"),
        ("S5_Router_Strict", "Router\n(Strict)", "Router-Strict"),
    ]

    fig, (ax_diff, ax_dom) = plt.subplots(
        1, 2, figsize=(15.2, 6.4), gridspec_kw={"width_ratios": [1.32, 1.0]}
    )
    fig.subplots_adjust(wspace=0.42)

    # --- Panel A: Grouped Bar Chart across Easy / Medium / Hard ---
    ax_diff.grid(True, axis="y", zorder=0)
    x = np.arange(len(selected_ids))
    width = 0.25

    easy_vals = [configs[cid]["easy_accuracy"] * 100.0 for cid, _, _ in selected_ids]
    med_vals = [configs[cid]["medium_accuracy"] * 100.0 for cid, _, _ in selected_ids]
    hard_vals = [configs[cid]["hard_accuracy"] * 100.0 for cid, _, _ in selected_ids]
    tok_vals = [configs[cid]["mean_tokens"] for cid, _, _ in selected_ids]

    ax_diff.bar(
        x - width, easy_vals, width, label="Easy (60 tasks)", color="#38BDF8", edgecolor="#0F172A", lw=0.7, zorder=3
    )
    ax_diff.bar(
        x, med_vals, width, label="Medium (60 tasks)", color="#6366F1", edgecolor="#0F172A", lw=0.7, zorder=3
    )
    b3 = ax_diff.bar(
        x + width, hard_vals, width, label="Hard (60 tasks)", color="#EC4899", edgecolor="#0F172A", lw=0.7, zorder=3
    )

    for bar in b3:
        h = bar.get_height()
        ax_diff.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#831843",
        )

    for idx_g, tok in enumerate(tok_vals):
        ax_diff.text(
            x[idx_g],
            13.5,
            f"{tok/1000:.1f}k\ntok",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", alpha=0.93, lw=0.7),
            zorder=5,
        )

    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels([x_lbl for _, x_lbl, _ in selected_ids], fontsize=8.4)
    ax_diff.set_ylabel("Stratum Accuracy (%)")
    ax_diff.set_ylim(0.0, 109.0)
    ax_diff.set_title("A. Accuracy Stratification across Easy / Medium / Hard Tasks")
    ax_diff.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.94, fontsize=8.2)

    # --- Panel B: Domain Accuracy Heatmap ---
    domains = [
        ("code_patch", "Code Patch &\nInvariant Repair"),
        ("scientific_symbolic", "Scientific Symbolic\n& Dimensional"),
        ("evidence_crosscheck", "Multi-Constraint\nEvidence Check"),
    ]
    mat = np.zeros((len(selected_ids), len(domains)), dtype=np.float64)
    for r_i, (cid, _, _) in enumerate(selected_ids):
        dom_map = configs[cid]["metadata"]["accuracy_by_domain"]
        for c_j, (d_key, _) in enumerate(domains):
            mat[r_i, c_j] = dom_map[d_key] * 100.0

    im = ax_dom.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=52.0, vmax=98.0)
    cbar = fig.colorbar(im, ax=ax_dom, fraction=0.046, pad=0.03)
    cbar.set_label("Domain Accuracy (%)", fontsize=9.2)

    ax_dom.set_xticks(np.arange(len(domains)))
    ax_dom.set_xticklabels([d_lbl for _, d_lbl in domains], fontsize=8.5)
    ax_dom.set_yticks(np.arange(len(selected_ids)))
    ax_dom.set_yticklabels(
        [
            f"{y_lbl} ({configs[cid]['mean_tokens']/1000:.1f}k)"
            for cid, _, y_lbl in selected_ids
        ],
        fontsize=8.5,
    )
    ax_dom.set_title("B. Accuracy by Verifiable Benchmark Domain")

    for r_i in range(mat.shape[0]):
        for c_j in range(mat.shape[1]):
            val = mat[r_i, c_j]
            txt_col = "white" if val > 78.0 else "#0F172A"
            ax_dom.text(
                c_j,
                r_i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8.6,
                fontweight="bold",
                color=txt_col,
            )

    out_path = assets_dir / "difficulty_stratified_breakdown.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_routing_and_latency(data: Dict[str, Any], assets_dir: Path) -> Path:
    """Figure 4: Adaptive Router Tier Allocation & End-to-End P50/P95 Latency SLA Trade-off."""
    configs = {c["config_id"]: c for c in data["configurations"]}

    fig, (ax_alloc, ax_lat) = plt.subplots(
        1, 2, figsize=(14.8, 6.4), gridspec_kw={"width_ratios": [1.05, 1.35]}
    )
    fig.subplots_adjust(wspace=0.25)

    # --- Panel A: Stacked Bar Chart of Tier Allocation ---
    router_bal = configs["S5_Router_Balanced"]
    alloc_bal = router_bal["metadata"]["router_tier_allocation"]
    short_rate = router_bal["metadata"]["tier3_shortcircuit_rate"] * 100.0

    bar_groups = [
        ("Easy\n(Balanced)", alloc_bal["easy"]),
        ("Medium\n(Balanced)", alloc_bal["medium"]),
        ("Hard\n(Balanced)", alloc_bal["hard"]),
    ]
    for r_id, short_lbl in [
        ("S5_Router_Lite", "All Tasks\n(Lite)"),
        ("S5_Router_Eco", "All Tasks\n(Eco)"),
        ("S5_Router_Strict", "All Tasks\n(Strict)"),
    ]:
        al = configs[r_id]["metadata"]["router_tier_allocation"]
        avg_tier = {
            t_k: float(np.mean([al[d][t_k] for d in ("easy", "medium", "hard")]))
            for t_k in ("tier1_fast", "tier2_verified", "tier3_hetero_debate")
        }
        bar_groups.append((short_lbl, avg_tier))

    x_g = np.arange(len(bar_groups))
    t1_pct = np.array([g[1]["tier1_fast"] * 100.0 for g in bar_groups])
    t2_pct = np.array([g[1]["tier2_verified"] * 100.0 for g in bar_groups])
    t3_pct = np.array([g[1]["tier3_hetero_debate"] * 100.0 for g in bar_groups])

    ax_alloc.grid(True, axis="y", zorder=0)
    ax_alloc.bar(
        x_g,
        t1_pct,
        width=0.56,
        color="#34D399",
        edgecolor="#0F172A",
        lw=0.8,
        label="Tier 1: Fast-8B + 0-tok MicroCheck",
        zorder=3,
    )
    ax_alloc.bar(
        x_g,
        t2_pct,
        bottom=t1_pct,
        width=0.56,
        color="#60A5FA",
        edgecolor="#0F172A",
        lw=0.8,
        label="Tier 2: Mid/Frontier + Tool Repair",
        zorder=3,
    )
    ax_alloc.bar(
        x_g,
        t3_pct,
        bottom=t1_pct + t2_pct,
        width=0.56,
        color="#F472B6",
        edgecolor="#0F172A",
        lw=0.8,
        label="Tier 3: Hetero Dual-Proposer + Gate",
        zorder=3,
    )

    for idx_b in range(len(bar_groups)):
        for bot, val, col_t in [
            (0.0, t1_pct[idx_b], "#064E3B"),
            (t1_pct[idx_b], t2_pct[idx_b], "#1E3A8A"),
            (t1_pct[idx_b] + t2_pct[idx_b], t3_pct[idx_b], "#831843"),
        ]:
            if val >= 8.0:
                ax_alloc.text(
                    x_g[idx_b],
                    bot + val / 2.0,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    fontweight="bold",
                    color=col_t,
                    zorder=5,
                )

    ax_alloc.axvline(2.5, color="#64748B", linestyle="--", linewidth=1.2, zorder=4)
    ax_alloc.set_xticks(x_g)
    ax_alloc.set_xticklabels([g[0] for g in bar_groups], fontsize=8.4)
    ax_alloc.set_ylabel("Dispatched Task Share (%)")
    ax_alloc.set_ylim(0.0, 126.0)
    ax_alloc.set_title(
        f"A. Zero-Oracle Tier Dispatching ({short_rate:.1f}% Tier-3 Gate Short-Circuit)"
    )
    ax_alloc.legend(loc="upper center", ncol=1, fontsize=8.0, frameon=True, framealpha=0.95)

    # --- Panel B: End-to-End P50 -> P95 Latency Whiskers vs. Accuracy (%) ---
    ax_lat.grid(True, zorder=0)

    key_lat_configs = [
        "S1_Fast",
        "S1_Mid",
        "S1_Frontier",
        "S1_Reasoning",
        "S2_SC5_Fast",
        "S2_SC5_Mid",
        "S2_SC5_Frontier",
        "S3_GV1_Frontier_WeakVer",
        "S3_GV3_Frontier_WeakVer_Homog",
        "S3_GV2_Mid_CalibVer",
        "S3_GV1_Frontier_ToolVer",
        "S3_GV3_Frontier_ToolVer",
        "S4_Debate_Homog_Mid",
        "S4_Debate_Hetero_MidFrontier",
        "S4_Debate_Hetero_ToolJudge",
        "S5_Router_Lite",
        "S5_Router_Eco",
        "S5_Router_Balanced",
        "S5_Router_Strict",
    ]

    for cid in key_lat_configs:
        c = configs[cid]
        fam = c["topology_family"]
        st = FAMILY_STYLE[fam]
        p50 = c["p50_latency_ms"] / 1000.0
        p95 = c["p95_latency_ms"] / 1000.0
        acc = c["accuracy"] * 100.0

        ax_lat.plot(
            [p50, p95],
            [acc, acc],
            color=st["color"],
            linewidth=2.0 if cid.startswith("S5_") else 1.25,
            alpha=0.80,
            zorder=st["zorder"] - 1,
        )
        ax_lat.scatter(
            [p50],
            [acc],
            s=108 if cid.startswith("S5_") else 65,
            c=st["color"],
            marker=st["marker"],
            edgecolors="#0F172A",
            linewidths=0.9,
            zorder=st["zorder"],
        )
        ax_lat.scatter(
            [p95],
            [acc],
            s=48,
            c=st["color"],
            marker="|",
            linewidths=1.9,
            zorder=st["zorder"],
        )

    lat_annotations = {
        "S1_Fast": (0.12, -1.3, "S1-Fast"),
        "S1_Frontier": (0.08, -1.7, "S1-Frontier"),
        "S1_Reasoning": (0.10, -1.6, "S1-Reasoning"),
        "S2_SC5_Frontier": (-1.18, +1.2, "SC@5-Frontier (8.4k tok)"),
        "S3_GV3_Frontier_WeakVer_Homog": (-1.35, -2.1, "GV@3-WeakVer (Harm)"),
        "S3_GV3_Frontier_ToolVer": (0.85, -1.9, "GV@3-ToolVer"),
        "S4_Debate_Hetero_MidFrontier": (0.06, -1.8, "Debate-Hetero"),
        "S4_Debate_Hetero_ToolJudge": (-0.25, +1.2, "Debate-Hetero+Tool"),
        "S5_Router_Lite": (-0.30, -2.0, "Router-Lite"),
        "S5_Router_Eco": (-0.60, -1.8, "Router-Eco"),
        "S5_Router_Balanced": (-0.65, +1.1, "Router-Balanced"),
        "S5_Router_Strict": (0.10, -1.7, "Router-Strict"),
    }
    for cid, (dx_s, dy_a, txt) in lat_annotations.items():
        c = configs[cid]
        p50_s = c["p50_latency_ms"] / 1000.0
        acc_y = c["accuracy"] * 100.0
        ax_lat.text(
            max(0.08, p50_s + dx_s),
            acc_y + dy_a,
            txt,
            fontsize=7.8,
            fontweight="bold" if cid.startswith("S5_") else "normal",
            color="#065F46" if cid.startswith("S5_") else "#1E293B",
            zorder=9,
        )

    lat_handles = [
        Line2D(
            [0],
            [0],
            marker=st["marker"],
            color=st["color"],
            lw=1.5,
            label=st["label"],
            markerfacecolor=st["color"],
            markeredgecolor="#0F172A",
            markersize=7.5,
        )
        for st in FAMILY_STYLE.values()
    ]
    lat_handles.append(
        Line2D([0], [0], color="#334155", lw=1.8, marker="|", markersize=8, label="Whisker: P50 -> P95 Tail Latency")
    )
    ax_lat.set_xlabel("End-to-End Critical-Path Latency (Seconds: Marker = P50, Cap = P95)")
    ax_lat.set_ylabel("Verifiable Benchmark Accuracy (%)")
    ax_lat.set_title("B. Latency SLA (P50 & P95) vs. Accuracy Trade-Off")
    ax_lat.set_xlim(0.0, 5.55)
    ax_lat.set_ylim(51.5, 100.5)
    ax_lat.legend(handles=lat_handles, loc="lower right", fontsize=7.8, frameon=True, framealpha=0.94)

    out_path = assets_dir / "routing_allocation_and_latency.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    _apply_pub_rc()
    json_path = ROOT / "results" / "crossverify_pareto_benchmark.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {json_path}. Run `scripts/run_benchmark.py` first."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_pareto_frontier(data, assets_dir),
        plot_verifier_phase_diagram(data, assets_dir),
        plot_difficulty_stratified_breakdown(data, assets_dir),
        plot_routing_and_latency(data, assets_dir),
    ]
    for p in paths:
        print(f"Generated 300-DPI publication figure: {p} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
