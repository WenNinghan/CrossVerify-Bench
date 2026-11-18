r"""Generate 300-DPI Publication Figures from 100% Real-API Multi-Family Live LLM Traces.

Reads ``results/live_llm_benchmark.json`` produced by ``scripts/run_live_llm_benchmark.py``
and generates two flagship figures in ``assets/``:
    1. ``assets/live_llm_pareto_frontier.png``: Empirical Token Cost vs. Accuracy Pareto
       frontier & difficulty-stratified accuracy across 5 real Chinese LLM families
       (Qwen, DeepSeek, GLM, MiMo, MiniMax) and 15 multi-agent configurations.
    2. ``assets/live_llm_verification_harm_audit.png``: Empirical Markov Verification Harm
       & Calibrated Gain Audit — plotting real measured $(\hat{\alpha}, \hat{\beta})$ points
       onto the analytical phase diagram + side-by-side $p_0 \to p_1$ transition breakdown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossverify.markov_theory import compute_phase_diagram_grid

FAMILY_STYLE = {
    "S1_SinglePass": {"color": "#3B6FB6", "marker": "o", "label": "S1: Single-Pass (Real LLM)"},
    "S2_SelfConsistency": {"color": "#E08214", "marker": "s", "label": "S2: Self-Consistency Voting (SC@K)"},
    "S3_GenVerifyLoop": {"color": "#7B52AB", "marker": "^", "label": "S3: Generator-Verifier Loop"},
    "S4_AdversarialDebateDAG": {"color": "#C93B5B", "marker": "D", "label": "S4: 3-Family Heterogeneous Debate"},
    "S5_ParetoAdaptiveRouter": {"color": "#0F8B6D", "marker": "P", "label": "S5: ParetoAdaptiveRouter (Ours)"},
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_live_pareto_frontier(data: Dict[str, Any], out_path: Path) -> None:
    cfgs = data["configurations"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.1), gridspec_kw={"width_ratios": [1.26, 1.0]})

    # --- Panel A: Empirical Token Cost vs Accuracy Pareto Frontier ---
    pareto_pts = sorted([c for c in cfgs if c["is_pareto_2d"]], key=lambda x: x["mean_tokens"])
    px = [p["mean_tokens"] for p in pareto_pts]
    py = [p["accuracy"] * 100.0 for p in pareto_pts]
    ax1.plot(px, py, color="#0F8B6D", lw=2.4, ls="-", alpha=0.88, zorder=3, label="Empirical Pareto Frontier")
    ax1.fill_between(px, py, min(c["accuracy"] * 100.0 for c in cfgs) - 3.0, color="#0F8B6D", alpha=0.06)

    seen_fams = set()
    for c in cfgs:
        fam = c["topology_family"]
        st = FAMILY_STYLE[fam]
        lbl = st["label"] if fam not in seen_fams else None
        seen_fams.add(fam)
        edge_col = "#D62728" if "[Harm]" in c["label"] else "#1F2937"
        sz = 155 if fam == "S5_ParetoAdaptiveRouter" else 95
        ax1.scatter(
            c["mean_tokens"],
            c["accuracy"] * 100.0,
            c=st["color"],
            marker=st["marker"],
            s=sz,
            edgecolors=edge_col,
            linewidths=1.5 if "[Harm]" in c["label"] else 0.9,
            zorder=6 if fam == "S5_ParetoAdaptiveRouter" else 4,
            label=lbl,
        )

    # Explicit non-overlapping label coordinates (x_text, y_text, short_label)
    explicit_labels = {
        "S1_DeepSeekFlash_NoThink": (620, 49.5, "DeepSeek-Flash (NoThink)"),
        "S1_QwenFlash_NoThink": (560, 66.8, "Qwen3.8-Flash (NoThink)"),
        "S2_SC5_DeepSeekFlash": (1820, 55.2, "SC@5 DeepSeek-Flash"),
        "S1_MiniMax_M31": (920, 79.2, "MiniMax-M3.1"),
        "S1_MiMo_V26": (260, 89.2, "MiMo-V2.6-Flash"),
        "S5_ParetoRouter_Eco": (230, 95.8, "ParetoRouter-Eco (Ours)"),
        "S1_GLM53_FlashX": (380, 102.6, "GLM-5.3-FlashX"),
        "S5_ParetoRouter_Strict": (3420, 101.2, "ParetoRouter-Strict (Ours)"),
        "S3_DeepSeek_ToolGLMVer": (2350, 83.2, "DeepSeek + MicroCheck + GLM Ver"),
        "S4_HeteroDebate_3Family": (4280, 97.8, "3-Family Debate DAG"),
        "S3_MiniMax_WeakQwenCritic": (3560, 75.0, "MiniMax + Weak Critic [Harm!]"),
        "S3_GLM53_WeakQwenCritic": (5350, 88.0, "GLM-5.3 + Weak Critic [Harm!]"),
    }
    cfg_map = {c["config_id"]: c for c in cfgs}
    for cid, (tx, ty, short_txt) in explicit_labels.items():
        if cid in cfg_map:
            pt = cfg_map[cid]
            col = "#B91C1C" if "Harm" in short_txt else ("#065F46" if "Ours" in short_txt else "#1E293B")
            ax1.annotate(
                f"{short_txt}\n({pt['accuracy']*100:.1f}%, {pt['mean_tokens']:.0f} tok)",
                xy=(pt["mean_tokens"], pt["accuracy"] * 100.0),
                xytext=(tx, ty),
                fontsize=7.9,
                fontweight="bold" if ("Ours" in short_txt or "Harm" in short_txt) else "normal",
                color=col,
                arrowprops={"arrowstyle": "->", "color": col, "lw": 0.85, "alpha": 0.72},
            )

    ax1.set_xlim(120, 7650)
    ax1.set_ylim(45.0, 107.8)
    ax1.set_xlabel("Mean Live API Token Consumption per Task (Input + Reasoning + Output)")
    ax1.set_ylabel("Exact Deterministic Oracle Accuracy (%)")
    ax1.set_title("A. Live Multi-Family LLM Cost–Accuracy Pareto Frontier (45 Tasks)")
    ax1.legend(loc="lower right", fontsize=8.2, framealpha=0.94)

    # --- Panel B: Stratified Easy / Medium / Hard Accuracy Across Key Strategies ---
    sel_ids = [
        ("S1_DeepSeekFlash_NoThink", "DeepSeek\nSinglePass"),
        ("S2_SC5_DeepSeekFlash", "DeepSeek\nSC@5 Vote"),
        ("S1_GLM53_FlashX", "GLM-5.3\nSinglePass"),
        ("S3_GLM53_WeakQwenCritic", "GLM-5.3 +\nWeak Critic"),
        ("S4_HeteroDebate_3Family", "3-Family\nDebate DAG"),
        ("S5_ParetoRouter_Eco", "Router-Eco\n(Ours)"),
        ("S5_ParetoRouter_Strict", "Router-Strict\n(Ours)"),
    ]
    sel_cfgs = [(cfg_map[cid], lbl) for cid, lbl in sel_ids if cid in cfg_map]
    x = np.arange(len(sel_cfgs))
    w = 0.25
    e_vals = [c["accuracy_by_difficulty"]["easy"] * 100.0 for c, _ in sel_cfgs]
    m_vals = [c["accuracy_by_difficulty"]["medium"] * 100.0 for c, _ in sel_cfgs]
    h_vals = [c["accuracy_by_difficulty"]["hard"] * 100.0 for c, _ in sel_cfgs]

    ax2.bar(x - w, e_vals, width=w, color="#60A5FA", edgecolor="#1E3A8A", lw=0.6, label="Easy (15)")
    ax2.bar(x, m_vals, width=w, color="#818CF8", edgecolor="#312E81", lw=0.6, label="Medium (15)")
    ax2.bar(x + w, h_vals, width=w, color="#0F8B6D", edgecolor="#064E3B", lw=0.6, label="Hard (15)")

    for idx, (c, _) in enumerate(sel_cfgs):
        ax2.text(
            idx,
            max(e_vals[idx], m_vals[idx], h_vals[idx]) + 1.8,
            f"Avg\n{c['accuracy']*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#111827",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in sel_cfgs], fontsize=8.0)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("Exact Oracle Accuracy (%)")
    ax2.set_title("B. Difficulty-Stratified Live Accuracy Breakdown")
    ax2.legend(loc="lower right", fontsize=8.4, framealpha=0.94)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_live_verification_harm_audit(data: Dict[str, Any], out_path: Path) -> None:
    audits: List[Dict[str, Any]] = data["markov_live_audits"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.9), gridspec_kw={"width_ratios": [1.06, 1.16]})

    # --- Panel A: Empirical (alpha_hat, beta_hat) Overlaid on Markov Phase Diagram ---
    alphas = np.linspace(0.0, 0.99, 120)
    betas = np.linspace(0.0, 0.99, 120)
    p0_ref, q_ic_ref, q_ci_ref = 0.80, 0.60, 0.16
    A, B, delta_p = compute_phase_diagram_grid(
        p0=p0_ref, q_i2c=q_ic_ref, q_c2i=q_ci_ref, alpha_grid=alphas, beta_grid=betas
    )
    Z = delta_p * 100.0
    norm = TwoSlopeNorm(vmin=min(-8.0, float(np.min(Z))), vcenter=0.0, vmax=max(8.0, float(np.max(Z))))
    pcm = ax1.pcolormesh(A, B, Z, cmap="RdYlGn", norm=norm, shading="auto", alpha=0.82)
    cbar = fig.colorbar(pcm, ax=ax1, pad=0.02)
    cbar.set_label(r"Theoretical Net Gain $\Delta p_1$ (pp, $p_0=0.80$ ref.)", fontsize=9.0)

    bound_alpha = np.clip(((1.0 - p0_ref) * q_ic_ref / (p0_ref * q_ci_ref)) * (1.0 - betas), 0.0, 1.0)
    ax1.plot(
        bound_alpha,
        betas,
        color="#111827",
        lw=2.2,
        ls="--",
        label=r"Critical Harm Boundary ($\Delta p_1 = 0$)",
    )

    box_positions = {
        "S3_DeepSeek_ToolGLMVer": (0.07, 0.09),
        "S3_MiniMax_ToolGLMVer": (0.07, 0.56),
        "S3_DeepSeek_WeakQwenCritic": (0.28, 0.24),
        "S3_GLM53_WeakQwenCritic": (0.43, 0.42),
        "S3_MiniMax_WeakQwenCritic": (0.46, 0.68),
    }

    for aud in audits:
        al = aud["alpha_false_rejection"]
        be = aud["beta_false_acceptance"]
        gain = aud["net_gain_pp"]
        is_neg = gain < 0
        mk = "X" if is_neg else "P"
        col = "#DC2626" if is_neg else "#047857"
        ax1.scatter(al, be, s=165, c=col, marker=mk, edgecolors="white", linewidths=1.5, zorder=8)
        short = (
            aud["config_id"]
            .replace("S3_", "")
            .replace("_WeakQwenCritic", " + WeakCritic")
            .replace("_ToolGLMVer", " + ToolGLMVer")
        )
        bx, by = box_positions.get(aud["config_id"], (0.35, 0.50))
        ax1.annotate(
            f"{short}\n(Δ={gain:+.1f} pp, α={al:.2f}, β={be:.2f})",
            xy=(al, be),
            xytext=(bx, by),
            fontsize=7.9,
            fontweight="bold",
            color="#7F1D1D" if is_neg else "#064E3B",
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": col, "alpha": 0.93},
            arrowprops={"arrowstyle": "->", "color": col, "lw": 1.1},
            zorder=9,
        )

    ax1.set_xlim(-0.03, 0.99)
    ax1.set_ylim(-0.03, 0.98)
    ax1.set_xlabel(r"Empirical Verifier Type-I False-Rejection Rate $\hat{\alpha} = P(\mathrm{reject} \mid \mathrm{Correct})$")
    ax1.set_ylabel(r"Empirical Type-II False-Acceptance Rate $\hat{\beta} = P(\mathrm{accept} \mid \mathrm{Incorrect})$")
    ax1.set_title("A. Real LLM Verifiers on Theoretical Markov Phase Space")
    ax1.legend(loc="upper right", fontsize=8.3, framealpha=0.94)

    # --- Panel B: Before vs After Verification Accuracy + Exact Markov Formula Alignment ---
    labels = [
        a["config_id"]
        .replace("S3_", "")
        .replace("_WeakQwenCritic", "\n+ Weak Qwen Critic")
        .replace("_ToolGLMVer", "\n+ MicroCheck & GLM Ver")
        for a in audits
    ]
    y_pos = np.arange(len(audits))
    h = 0.27
    p0_v = [a["p0_empirical"] * 100.0 for a in audits]
    p1_v = [a["p1_empirical"] * 100.0 for a in audits]
    pt_v = [a["p1_markov_formula"] * 100.0 for a in audits]

    ax2.barh(y_pos + h, p0_v, height=h, color="#94A3B8", edgecolor="#334155", label=r"Initial Single-Pass $\hat{p}_0$")
    bar_colors = ["#EF4444" if a["net_gain_pp"] < 0 else "#10B981" for a in audits]
    ax2.barh(y_pos, p1_v, height=h, color=bar_colors, edgecolor="#1F2937", label=r"Post-Verification Live $\hat{p}_1$ (Red=Harm, Green=Gain)")
    ax2.scatter(pt_v, y_pos, color="#111827", marker="D", s=50, zorder=7, label=r"Closed-Form Theorem 1 Identity $p_1^{\mathrm{theory}}$")

    for i, a in enumerate(audits):
        delta = a["net_gain_pp"]
        tag = f"{delta:+.1f} pp ({a['n_corrupted_by_critic']} corrupted, {a['n_recovered_by_critic']} fixed)"
        ax2.text(
            max(p0_v[i], p1_v[i]) + 1.4,
            y_pos[i],
            tag,
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color="#B91C1C" if delta < 0 else "#047857",
        )

    ax2.set_yticks(y_pos + h / 2)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlim(0, 144)
    ax2.set_ylim(-0.45, len(audits) + 0.38)
    ax2.set_xlabel("Exact Deterministic Oracle Accuracy (%)")
    ax2.set_title("B. Empirical Verification Harm vs. Tool-Grounded Gain (45 Tasks)")
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.94, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    json_path = ROOT / "results" / "live_llm_benchmark.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _apply_style()
    plot_live_pareto_frontier(payload, ROOT / "assets" / "live_llm_pareto_frontier.png")
    plot_live_verification_harm_audit(payload, ROOT / "assets" / "live_llm_verification_harm_audit.png")
